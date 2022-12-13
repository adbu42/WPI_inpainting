import os
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from data.dataloader import GetData
from loss.InpaintingLoss import InpaintingLossWithGAN
from models.LBAMModel import LBAMModel, VGG16FeatureExtractor
import yaml


parser = argparse.ArgumentParser()
parser.add_argument('--numOfWorkers', type=int, default=4,
                    help='workers for dataloader')
parser.add_argument('--modelsSavePath', type=str, default='',
                    help='path for saving models')
parser.add_argument('--logPath', type=str,
                    default='')
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--loadSize', type=int, default=350,
                    help='image loading size')
parser.add_argument('--cropSize', type=int, default=256,
                    help='image training size')
parser.add_argument('--dataRoot', type=str,
                    default='')
parser.add_argument('--maskRoot', type=str,
                    default='')
parser.add_argument('--pretrained', type=str, default='', help='pretrained models for finetuning')
parser.add_argument('--train_epochs', type=int, default=500, help='training epochs')
args = parser.parse_args()


config_path = args.config_path
with open(config_path) as f:
    config = yaml.safe_load(f)


batchSize = args.batchSize
loadSize = (args.loadSize, args.loadSize)
cropSize = (args.cropSize, args.cropSize)

if not os.path.exists(args.modelsSavePath):
    os.makedirs(args.modelsSavePath)

dataRoot = args.dataRoot
maskRoot = args.maskRoot

imgData = GetData(dataRoot, maskRoot, loadSize, cropSize)
data_loader = DataLoader(imgData, batch_size=batchSize,
                         shuffle=True, num_workers=args.numOfWorkers, drop_last=False, pin_memory=True)

num_epochs = args.train_epochs

netG = LBAMModel(4, 3)
if args.pretrained != '':
    netG.load_state_dict(torch.load(args.pretrained))

count = 1


if cuda:
    criterion = criterion.cuda()

    if numOfGPUs > 1:
        criterion = nn.DataParallel(criterion, device_ids=range(numOfGPUs))

print('OK!')

for i in range(1, num_epochs + 1):
    netG.train()

    for inputImgs, GT, masks in (data_loader):


        netG.zero_grad()

        fake_images = netG(inputImgs, masks)
        G_loss = criterion(inputImgs[:, 0:3, :, :], masks, fake_images, GT, count, i)
        G_loss = G_loss.sum()
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        print('Generator Loss of epoch{} is {}'.format(i, G_loss.item()))

        count += 1

        """ if (count % 4000 == 0):
            torch.save(netG.module.state_dict(), args.modelsSavePath +
                    '/Places_{}.pth'.format(i)) """

    if (i % 10 == 0):
        if numOfGPUs > 1:
            torch.save(netG.module.state_dict(), args.modelsSavePath +
                       '/LBAM_{}.pth'.format(i))
        else:
            torch.save(netG.state_dict(), args.modelsSavePath +
                       '/LBAM_{}.pth'.format(i))