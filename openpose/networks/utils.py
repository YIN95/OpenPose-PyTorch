# -*- coding:utf-8 -*-
"""
Author:
    Wenjie Yin, yinw@kth.se
    Reference: https://github.com/Hzzone/pytorch-openpose
"""

import torch.nn as nn
from collections import OrderedDict

def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(
                in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, conv2d))

            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))
    
    return nn.Sequential(OrderedDict(layers))