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
import cv2


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
from time import *

torch.multiprocessing.set_sharing_strategy('file_system')
import math


def psnr1(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def psnr2(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

#print args
def print_setting(args):
    import time
    print(args)
    time.sleep(5)

import glob
import re
import os


def ssim_and_save(com_data,cln_data):
    '''
    :param com_data: [N,C,H,W] | np.array [0,1]
    :param cln_data: [N,C,H,W] | np.array [0,1]
    :return:
    '''
    com_data = ((np.transpose(com_data,[0,2,3,1]))*255).astype(np.float32)
    cln_data = ((np.transpose(cln_data,[0,2,3,1]))*255).astype(np.float32) # only np.float32 is supported
    ssim_list=[]
    psnr_list = []
    for index in range(com_data.shape[0]):
        com_img = com_data[index]
        cln_img = cln_data[index]
        psnr = psnr2(com_img,cln_img)
        psnr_list.append(psnr)
        grayA = cv2.cvtColor(com_img, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(cln_img, cv2.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        ssim_list.append(score)
        print("img index: {} , SSIM: {} PSNR:{}".format(index,score,psnr))
        base_dir = os.path.join("defend_image",args.defence_method)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        com_filename = os.path.join(base_dir,str(index)+".png")
        cln_filename = os.path.join("clean_image",str(index)+".png")
        cv2.imwrite(com_filename,com_data[index])
        cv2.imwrite(cln_filename,cln_data[index])
    print("avg ssim:{} , avg psnr:{} ".format(np.asarray(ssim_list).mean(),np.asarray(psnr_list).mean()))

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
    # file_name = "cifar10_vgg16_model_299.pth"
    model = torch.load(os.path.join(save_dir, file_name))
    model = model['net']
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.eval()
    clncorrect_nodefence = 0
    clncorrect_defence = 0
    correct_defence = 0
    for clndata, target in test_loader:
        clndata, target = clndata.to(device), target.to(device)

        # # clean data without defence
        with torch.no_grad():
            output = model(clndata.float())
        pred = output.max(1, keepdim=True)[1]
        clncorrect_nodefence += pred.eq(target.view_as(pred)).sum().item()

        # defence
        begin_time = time()
        defence_data = defencer(adv_data=clndata.cpu().numpy(),defence_method=args.defence_method, clip_values=(0,1), eps=args.epsilon,bit_depth=8, apply_fit=False, apply_predict=True)
        end_time = time()
        run_time = end_time - begin_time
        print('该循环程序运行时间：', run_time)
        if args.test_ssim:
            print("测试ssim值")
            ssim_and_save(defence_data,clndata.cpu().numpy())
            break
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
    
    # clean test
    test_loader = get_handled_cifar10_test_loader(batch_size=50, num_workers=2, shuffle=False,nb_samples=10000)
    correct = 0
    for clndata, target in test_loader:
        clndata, target = clndata.to(device), target.to(device)
        # # clean data without defence
        with torch.no_grad():
            output = model(clndata.float())
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        print('\ncln Test set: '
              ' cln acc: {}/{} ({:.0f}%) \n'.format(
                   correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))