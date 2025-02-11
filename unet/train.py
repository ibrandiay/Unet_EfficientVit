"""
Training code for the UNet segmentation with the following features:
- Cross Entropy Loss
- Adam Optimizer
- Learning Rate Scheduler
- Multi GPU Training
- Tensorboard Logging
- Model Checkpointing
- Model Loading

Usage:
    python train.py --epochs 5 --batch-size 2 --lr 0.01 --load <path to checkpoint> --scale 0.5 --validation 10

Warning: this code is build to run on a multi GPU machine. If you want to run it on a single GPU machine, you need to change the code.
         change the line: net = nn.DataParallel(net) to net = net
         and remove the .module in the code.


Author: Ibra Ndiaye
date: 10/11/2022

"""
import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet, BasicDataset

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

dir_img = 'data/images_crop/'
dir_mask = 'data/mask_crop/'
dir_checkpoint = 'checkpoints/'


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              n_classes=2,
              class_weights=[1, 1], ):
    dataset = BasicDataset(dir_img, dir_mask, img_scale, n_classes)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    if net.module.n_classes > 1:
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device=device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.module.n_channels, \
                    f'Network has been defined with {net.module.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.module.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks.squeeze(1))
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (train)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                # validation
        if epoch % 10 == 0 and epoch != 0:
            val_score = eval_net(net, val_loader, device, n_val)
            logging.info('Validation cross entropy: {}'.format(val_score))
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            if not epoch % 10:
                torch.save(net.state_dict(),
                           dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=500,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-w', '--weights', nargs='+', type=float, default=[1, 1, 1, 1],
                        help='Class weights to use in loss calculation')
    parser.add_argument('--n_classes', type=int, default=4,
                        help='Number of classes in the segmentation')
    parser.add_argument('--n_channels', type=int, default=3,
                        help='Number of channels in input images')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=5,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N

    n_classes = args.n_classes
    n_channels = args.n_channels
    class_weights = np.array(args.weights).astype(np.float)
    assert len(class_weights) == n_classes, \
        'Lenght of the weights-vector should be equal to the number of classes'
    # tranfer learning from pretrained model model.pth
    net = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=True)

    if torch.cuda.device_count() > 1:
        print("Utilisons", torch.cuda.device_count(), "GPU !")
        net = torch.nn.DataParallel(net, device_ids=[0, 1])

    logging.info(f'Network:\n'
                 f'\t{net.module.n_channels} input channels\n'
                 f'\t{net.module.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.module.bilinear else "Dilated conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  n_classes=4,
                  class_weights=class_weights,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
