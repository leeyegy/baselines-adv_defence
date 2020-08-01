import  numpy as np
from art.defences.preprocessor import FeatureSqueezing,PixelDefend,ThermometerEncoding,TotalVarMin,JpegCompression,SpatialSmoothing
from art.classifiers import Classifier
from art.classifiers import PyTorchClassifier
import torch.nn as nn
import  torch
from pixel_cnn import *
from torch import nn, optim, cuda, backends
from new_pixel_cnn import  *



def defencer(adv_data, defence_method, clip_values, eps=16,bit_depth=8, apply_fit=False, apply_predict=True):
    '''
    :param adv_data: np.ndarray | [N C H W ]
    :param defence_method: | str
    :param clip_values:Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features. | `tuple`
    :param bit_depth: The number of bits per channel for encoding the data. | 'int'
    :param apply_fit:  True if applied during fitting/training. | bool
    :param apply_predict: True if applied during predicting. | bool
    :return: defended data | np.ndarray | [N C H W]
    '''

    # step 1. define a defencer
    if defence_method == "FeatureSqueezing":
        defence = FeatureSqueezing(clip_values=clip_values, bit_depth=bit_depth, apply_fit=apply_fit, apply_predict=apply_predict)
    elif defence_method == "PixelDefend":
        criterion = nn.CrossEntropyLoss()
        # fm = 64
        # pixel_cnn_model = nn.Sequential(
        #     MaskedConv2d('A', 3, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        #     MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        #     MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        #     MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        #     MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        #     MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        #     MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        #     MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        #     nn.Conv2d(fm, 256, 1))
        pixel_cnn_model = Pixel_cnn_net().cuda()
        pixel_cnn_model = torch.load("models/pixel_cnn_epoch_29.pth")
        # pixel_cnn_model = PixelCNN().cuda()
        # print(pixel_cnn_model)
        optimizer = optim.Adam(pixel_cnn_model.parameters())
        pixel_cnn = PyTorchClassifier(
            model=pixel_cnn_model,
            clip_values=(0, 1),
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, 32, 32),
            nb_classes=10,
        )
        defence =PixelDefend(clip_values=clip_values, eps=eps, pixel_cnn=pixel_cnn, apply_fit=apply_fit, apply_predict=apply_predict)
        adv_data = np.transpose(adv_data,[0,3,2,1])
    elif defence_method == "ThermometerEncoding":
        defence = ThermometerEncoding(clip_values=clip_values)
    elif defence_method == "TotalVarMin":
        defence = TotalVarMin(clip_values=clip_values)
    elif defence_method == "JPEGCompression":
        defence = JpegCompression(clip_values=clip_values)
    elif defence_method == "SpatialSmoothing":
        defence = SpatialSmoothing(clip_values=clip_values)

    adv_data = np.transpose(adv_data,[0,3,2,1])
    # step2. defend
    # print(adv_data.shape)
    res = defence(adv_data)[0]
    res = np.transpose(res,[0,3,2,1])
    # print(res.shape)
    return res

    # if defence_method=="FeatureSqueezing":
    #     return  res
    # else:
    #     return np.transpose(res,[0,3,2,1])



