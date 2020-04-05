

from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import TRAINED_MODEL_PATH
from advertorch.attacks import CarliniWagnerL2Attack,GradientSignAttack,L2PGDAttack,SpatialTransformAttack,JacobianSaliencyMapAttack,MomentumIterativeAttack

from skimage.io import imread, imsave
from skimage.measure import compare_psnr, compare_ssim


import numpy as np
import matplotlib
import  time

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_generator import get_handled_cifar10_train_loader,get_handled_cifar10_test_loader,get_test_adv_loader
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
import os, glob, datetime, time

import sys
from torch.autograd import Variable
sys.path.append("../../")
from networks import *
from defence import defencer
from pixel_cnn import *#没有这行代码的话，torch.load 来加载pixel_cnn会报错
from config import  args

torch.multiprocessing.set_sharing_strategy('file_system')






#print args
def print_setting(args):
    import time
    print(args)
    time.sleep(5)

import glob
import re
import os


if __name__ == '__main__':
    print_setting(args)

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # data
    test_loader = get_test_adv_loader(attack_method=args.attack_method,epsilon=args.epsilon)


    # load net
    save_dir = "checkpoint"
    file_name = 'wide-resnet-' + str(args.depth) + 'x' + str(args.widen_factor) + '.t7'
    model = torch.load(os.path.join(save_dir, file_name))
    model = model['net']
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    nb_epoch = 1
    for epoch in range(nb_epoch):
        model.eval()
        clncorrect_nodefence = 0
        clncorrect_defence = 0
        correct_defence = 0

        sum_fosc_loss = 0
        for clndata, target in test_loader:
            clndata, target = clndata.to(device), target.to(device)

            # # clean data without defence
            with torch.no_grad():
                output = model(clndata.float())
            pred = output.max(1, keepdim=True)[1]
            clncorrect_nodefence += pred.eq(target.view_as(pred)).sum().item()

            # defence
            defence_data = defencer(adv_data=clndata.cpu().numpy(),defence_method=args.defence_method, clip_values=(0,1), eps=args.epsilon,bit_depth=8, apply_fit=False, apply_predict=True)
            defence_data = torch.from_numpy(defence_data).to(device)

            with torch.no_grad():
                output = model(defence_data.float())
            pred = output.max(1, keepdim=True)[1]
            correct_defence += pred.eq(target.view_as(pred)).sum().item()


        print('\nadv Test set: '
              ' adv acc: {}/{} ({:.0f}%) \n'.format(
                   clncorrect_nodefence, len(test_loader.dataset),
                  100. * clncorrect_nodefence / len(test_loader.dataset)))



        print(' defence test: '
              '  acc: {}/{} ({:.0f}% )\n'.format(
                   correct_defence, len(test_loader.dataset),
                  100. * correct_defence / len(test_loader.dataset)))

