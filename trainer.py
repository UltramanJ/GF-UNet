import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import DiceLoss,test_single_volume
import numpy as np


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_val = Synapse_dataset(base_dir='/storage/SA/Swin-Unet-main/data/test_vol_h5', list_dir=args.list_dir, split="test",
                             )
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-5)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = 10e10
    best_performance = 0
    eval_interval = args.eval_interval
    for epoch_num in iterator:
        model.train()
        batch_dice_loss = 0
        batch_ce_loss = 0
        for i_batch, sampled_batch in tqdm(enumerate(train_loader), desc=f"Train: {epoch_num}", total=len(train_loader),
                                           leave=False):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            lr_ = base_lr
            #lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            # logging.info('Train: iteration : %d/%d, lr : %f, loss : %f, loss_ce: %f, loss_dice: %f' % (
            #     iter_num, epoch_num, lr_, loss.item(), loss_ce.item(), loss_dice.item()))
            batch_dice_loss += loss_dice.item()
            batch_ce_loss += loss_ce.item()
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        batch_ce_loss /= len(train_loader)
        batch_dice_loss /= len(train_loader)
        batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
        logging.info('Train epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
            epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))
        if batch_loss < best_loss:
            save_mode_path = os.path.join(snapshot_path, 'best2_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            best_loss = batch_loss
            logging.info("Saved new best model")

            # 定期保存last模型
        if (epoch_num + 1) % 10 == 0:
            save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("Saved periodic last model")
        if epoch_num >= int((max_epoch) / 2) and (epoch_num + 1) % eval_interval == 0:
            model.eval()

            metric_list = 0.0
            for i_batch, sampled_batch in enumerate(val_loader):
                image, label = sampled_batch["image"], sampled_batch["label"]
                metric_i = test_single_volume(image, label, model, classes=num_classes,
                                              patch_size=[args.img_size, args.img_size])
                metric_list += np.array(metric_i)
            metric_list = metric_list / len(db_val)
            for class_i in range(num_classes - 1):
                writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                  metric_list[class_i,0], iter_num)
                writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                                   metric_list[class_i, 1], iter_num)

            performance = np.mean(metric_list, axis=0)[0]

            mean_hd95 = np.mean(metric_list, axis=0)[1]
            writer.add_scalar('info/val_mean_dice', performance, iter_num)
            writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

            if performance > best_performance:
                best_iteration, best_performance = iter_num, performance
                save_best = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), save_best)
                logging.info('Best model | iteration %d : mean_dice : %f ' % (
                    iter_num, performance))

            logging.info('iteration %d : mean_dice : %f ' % (iter_num, performance))
            model.train()
    writer.close()
    return "Training Finished!"
