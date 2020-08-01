from BPDA import  BPDAattack
import  numpy as np
import torch
import argparse
from art.defences.preprocessor import FeatureSqueezing,PixelDefend,ThermometerEncoding,TotalVarMin,JpegCompression,SpatialSmoothing
from art.classifiers import Classifier
from art.classifiers import PyTorchClassifier
import torch.nn as nn
from data_generator import  get_handled_cifar10_test_loader

def _JpegCompression(data):
    '''
    :param data: tensor.cuda() | [N,C,H,W] | [0,1]
    :return: tensor.cuda() | [N,C,H,W] | [0,1]
    '''
    # defence
    data = np.transpose(data.cpu().numpy(),[0,3,2,1])
    res = JpegCompression(clip_values=(0, 1))(data)[0]
    return torch.from_numpy(np.transpose(res,[0,3,2,1])).cuda()

def _SpatialSmoothing(data):
    '''
    :param data: tensor.cuda() | [N,C,H,W] | [0,1]
    :return: tensor.cuda() | [N,C,H,W] | [0,1]
    '''
    # defence
    data = np.transpose(data.cpu().numpy(),[0,3,2,1])
    res = SpatialSmoothing(clip_values=(0, 1))(data)[0]
    return torch.from_numpy(np.transpose(res,[0,3,2,1])).cuda()


def main(args):
    # load data
    testLoader = get_handled_cifar10_test_loader(batch_size=50, num_workers=2, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    print('| Resuming from checkpoint...')
    checkpoint = torch.load('./checkpoint/wide-resnet-28x10.t7') # for cifar10
    model = checkpoint['net']
    model = model.to(device)

    # define defence_method
    defence = _JpegCompression if args.defence_method == "JPEGCompression" else _SpatialSmoothing

    # define adversary
    adversary = BPDAattack(model, defence, device,
                                epsilon=args.epsilon,
                                learning_rate=0.01,
                                max_iterations=args.max_iterations,test=test)

    # model test
    model.eval()
    clncorrect_nodefence = 0
    for data,target in testLoader:
        data, target = data.cuda(), target.cuda()
        # attack
        adv_data = adversary.perturb(data,target)
        # defence
        denoised_data = defence(adv_data)
        with torch.no_grad():
            output = model(denoised_data.float())
        pred = output.max(1, keepdim=True)[1]
        pred = pred.double()
        target = target.double()
        clncorrect_nodefence += pred.eq(target.view_as(pred)).sum().item()  # item： to get the value of tensor
    print('\nTest set with feature-dis defence against BPDA'
              ' cln acc: {}/{} ({:.0f}%)\n'.format(
                clncorrect_nodefence, len(testLoader.dataset),
                  100. * clncorrect_nodefence / len(testLoader.dataset)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--epsilon",type=float,default=8/255)

    # defence
    parser.add_argument("--defence_method",default="JPEGCompression",type =str,choices=['JPEGCompression',"SpatialSmoothing"])

    # BPDA ATTACK
    parser.add_argument("--max_iterations",default=10,type =int)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    main(args)