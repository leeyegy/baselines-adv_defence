

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
from data_generator import get_handled_cifar10_train_loader,get_handled_cifar10_test_loader
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
import os, glob, datetime, time

import sys
from torch.autograd import Variable
sys.path.append("../../")
from networks import *
from defence import defencer


torch.multiprocessing.set_sharing_strategy('file_system')



def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))


def show(img_tensor):
    transforms.ToPILImage()(img_tensor).show()

# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(args.num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, args.num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, args.num_classes)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, args.num_classes)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name

#print args
def print_setting(args):
    import time
    print(args)
    time.sleep(5)

import glob
import re
import os
def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,"*model_*.pth"))
    epoch = []
    if file_list:
        for file in file_list:
            result = re.findall("model_(.*).pth.*",file)
            if result:
                epoch.append(int(result[0]))
        if epoch:
            return max(epoch)
    return 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', default="adv", help="cln | adv")
    parser.add_argument('--train_batch_size', default=50, type=int)
    parser.add_argument('--test_batch_size', default=1000, type=int)
    parser.add_argument('--log_interval', default=200, type=int)
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    parser.add_argument('--monitor', default=False, type=bool, help='if monitor the training process')
    parser.add_argument('--start_save', default=90, type=int, help='the threshold epoch which will start to save imgs data using in testing')

    # attack
    parser.add_argument("--attack_method", default="PGD", type=str,
                        choices=['FGSM', 'PGD','Momentum','STA'])
    parser.add_argument('--targeted', action='store_true',help='if to minimize')


    parser.add_argument('--epsilon', type = float,default=8/255, help='if pd_block is used')

    #model save
    parser.add_argument('--des', default='targeted', type=str, help='for model saving and loading ')


    #resume CNN
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/MNIST]')

    # ddid
    parser.add_argument('--sigma', type = int,default=30, help='for ddid ')


    #net
    parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
    parser.add_argument('--depth', default=28, type=int, help='depth of model')
    parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
    parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
    parser.add_argument('--num_classes', default=10, type=int)

    # test
    parser.add_argument('--test_samples', default=100, type=int)

    # defence
    parser.add_argument('--defence_method', default="FeatureSqueezing", type=str)


    args = parser.parse_args()
    print_setting(args)

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.mode == "cln":
        flag_advtrain = False
    elif args.mode == "adv":
        flag_advtrain = True
    else:
        raise

    # data
    # train_loader = get_handled_cifar10_train_loader(num_workers=4,shuffle=True,batch_size=args.train_batch_size)
    test_loader = get_handled_cifar10_test_loader(num_workers=4,shuffle=False,batch_size=args.train_batch_size,nb_samples=args.test_samples)


    # load net
    save_dir = "checkpoint"
    file_name = 'wide-resnet-' + str(args.depth) + 'x' + str(args.widen_factor) + '.t7'
    model = torch.load(os.path.join(save_dir, file_name))
    model = model['net']
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    if flag_advtrain:
        from advertorch.attacks import LinfPGDAttack
        if args.attack_method == "PGD":
            adversary = LinfPGDAttack(
                model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.epsilon,
                nb_iter=20, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
                targeted=args.targeted)
        else:
            pass
        # elif args.attack_method == "FGSM":
        #     adversary =GradientSignAttack(
        #         model,loss_fn=FOSCLoss(),
        #         clip_min=0.0, clip_max=1.0,eps=0.007,targeted=False)
        # elif args.attack_method == "Momentum":
        #     adversary =MomentumIterativeAttack(
        #         model, loss_fn=FOSCLoss(), eps=args.epsilon,
        #         nb_iter=40, decay_factor=1.0, eps_iter=1.0, clip_min=0.0, clip_max=1.0,
        #         targeted=False,ord=np.inf)
        # elif args.attack_method == "STA":
        #     adversary =SpatialTransformAttack(
        #         model,num_classes=args.num_classes, loss_fn=FOSCLoss(),
        #         initial_const=0.05, max_iterations=1000, search_steps=1, confidence=0, clip_min=0.0, clip_max=1.0,
        #         targeted=False,abort_early=True )
    time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    nb_epoch = 1
    for epoch in range(nb_epoch):
        #evaluate
        model.eval()
        clncorrect_nodefence = 0
        clncorrect_defence = 0


        if flag_advtrain:
            advcorrect = 0
            advcorrect_nodefence = 0
            correct_defence = 0

        sum_fosc_loss = 0
        for clndata, target in test_loader:
            clndata, target = clndata.to(device), target.to(device)
            # # clean data without defence
            with torch.no_grad():
                output = model(clndata.float())
            pred = output.max(1, keepdim=True)[1]
            clncorrect_nodefence += pred.eq(target.view_as(pred)).sum().item()

            # # defence
            # defence_clndata = ddid_batch(clndata, (args.sigma / 255) ** 2)
            # with torch.no_grad():
            #     output = model(defence_clndata.float())
            # pred = output.max(1, keepdim=True)[1]
            # clncorrect_defence += pred.eq(target.view_as(pred)).sum().item()


            if flag_advtrain:
                with ctx_noparamgrad_and_eval(model):
                    advdata = adversary.perturb(clndata,target)


                # no defence
                with torch.no_grad():
                    output = model(advdata.float())
                pred = output.max(1, keepdim=True)[1]
                advcorrect_nodefence += pred.eq(target.view_as(pred)).sum().item()

                # defence
                defence_data = defencer(adv_data=advdata.cpu().numpy(),defence_method=args.defence_method, clip_values=(0,1), bit_depth=8, apply_fit=False, apply_predict=True)
                defence_data = torch.from_numpy(defence_data).to(device)

                with torch.no_grad():
                    output = model(defence_data.float())
                pred = output.max(1, keepdim=True)[1]
                correct_defence += pred.eq(target.view_as(pred)).sum().item()

        # print('\nclean Test set: '
        #       ' cln acc: {}/{} ({:.0f}%) cln_defence acc: {}/{} (:.0f%)\n'.format(
        #            clncorrect_nodefence, len(test_loader.dataset),
        #           100. * clncorrect_nodefence / len(test_loader.dataset),clncorrect_defence,len(test_loader.dataset),100.*clncorrect_defence/len(test_loader.dataset)))

        print('\nclean Test set: '
              ' cln acc: {}/{} ({:.0f}%) \n'.format(
                   clncorrect_nodefence, len(test_loader.dataset),
                  100. * clncorrect_nodefence / len(test_loader.dataset)))


        if flag_advtrain:
            print('adv Test set: '
                  ' adv acc: {}/{} ({:.0f}% )\n'.format(
                       advcorrect_nodefence, len(test_loader.dataset),
                      100. * advcorrect_nodefence / len(test_loader.dataset)))
            print(' defence test: '
                  ' adv acc: {}/{} ({:.0f}% )\n'.format(
                       correct_defence, len(test_loader.dataset),
                      100. * correct_defence / len(test_loader.dataset)))

