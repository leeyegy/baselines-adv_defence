#! /usr/bin/env python
# codes are based on jzbontar/pixelcnn-pytorch
import os
import time
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.autograd import Variable
# from torch.utils import data
from torchvision import datasets, transforms, utils
backends.cudnn.benchmark = True
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10




class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class Pixel_cnn_net(nn.Module):
    def __init__(self):
        super(Pixel_cnn_net,self).__init__()
        layers = []
        fm = 64
        for i in range(8):
            if i ==0:
                layers.append(MaskedConv2d('A', 3,  fm, 7, 1, 3, bias=False))
            else:
                layers.append(MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False))
            layers.append(nn.BatchNorm2d(fm))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(fm, 256, 1))
        self.pixel_cnn = nn.Sequential(*layers)

    def forward(self,x):
            return self.pixel_cnn(x)


#
# fm = 64
# net = nn.Sequential(
#     MaskedConv2d('A', 3,  fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
#     MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
#     MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
#     MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
#     MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
#     MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
#     MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
#     MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
#     nn.Conv2d(fm, 256, 1))
# print (net)
def findLastCheckpoint():
    import glob,re,os
    file_list = glob.glob(os.path.join("models","pixel_cnn_epoch_*.pth"))
    epoch = []
    if file_list:
        for file in file_list:
            result = re.findall(".*pixel_cnn_epoch_(.*).pth",file)
            epoch.append(int(result[0]))
        return max(epoch)
    return -1

if __name__ == '__main__':
    net = Pixel_cnn_net().cuda()


    tr = DataLoader(CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor()),
                         batch_size=50, shuffle=True, num_workers=4, pin_memory=True)
    te = DataLoader(CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor()),
                         batch_size=50, shuffle=False, num_workers=4, pin_memory=True)

    sample = torch.Tensor(144, 1, 28, 28).cuda()
    optimizer = optim.Adam(net.parameters())

    init_epoch = findLastCheckpoint()
    if init_epoch >=0:
        print("restored from epoch:{}".format(init_epoch+1))
        net = torch.load("models/pixel_cnn_epoch_"+str(init_epoch)+".pth")


    for epoch in range(init_epoch+1,30):
        # train
        err_tr = []
        cuda.synchronize()
        time_tr = time.time()
        net.train(True)
        for input, _ in tr:
            input = Variable(input.cuda(async=True))
            target = Variable((input.data[:,0] * 255).long())
            loss = F.cross_entropy(net(input), target)
            # print(loss.item())
            err_tr.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        cuda.synchronize()
        time_tr = time.time() - time_tr

        # compute error on test set
        err_te = []
        cuda.synchronize()
        time_te = time.time()
        net.train(False)
        for input, _ in te:
            input = Variable(input.cuda(async=True), volatile=True)
            target = Variable((input.data[:,0] * 255).long())
            loss = F.cross_entropy(net(input), target)
            err_te.append(loss.item())
        cuda.synchronize()
        time_te = time.time() - time_te



        # # sample
        # sample.fill_(0)
        # net.train(False)
        # for i in range(28):
        #     for j in range(28):
        #         out = net(Variable(sample, volatile=True))
        #         probs = F.softmax(out[:, :, i, j]).data
        #         sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.
        # utils.save_image(sample, 'sample_{:02d}.png'.format(epoch), nrow=12, padding=0)

        print('epoch={}; nll_tr={:.7f}; nll_te={:.7f}; time_tr={:.1f}s; time_te={:.1f}s'.format(
            epoch, np.mean(err_tr), np.mean(err_te), time_tr, time_te))

        torch.save(net,"models/pixel_cnn_epoch_"+str(epoch)+".pth")
