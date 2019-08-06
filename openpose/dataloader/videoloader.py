# -*- coding:utf-8 -*-
"""
Author:
    Wenjie Yin, yinw@kth.se
    Reference: https://github.com/Hzzone/pytorch-openpose
"""

import torch
import os
import cv2

from torch.utils.data import Dataset, DataLoader,random_split,Sampler


class VideoData(Dataset):
    def __init__(self):
        super(VideoData, self).__init__()

        # data_path = 