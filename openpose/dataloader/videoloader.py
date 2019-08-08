'''
@Author: Wenjie Yin, yinwenjie159@hotmail.com
@Date: 2019-08-08 18:29:06
@LastEditors: Wenjie Yin, yinwenjie159@hotmail.com
@LastEditTime: 2019-08-08 18:41:54
@Description: file content
'''
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
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import PIL.Image as Image

def get_transform(mode='default'):
    if mode in ['default']:
        return transforms.Compose([
            transforms.ToTensor()
    ])

class ImagesData(Dataset):
    def __init__(self, data_path, transfrom=None):
        super(ImagesData, self).__init__()

        self.data_path = data_path
        self.images = os.listdir(data_path)
        self.transfrom = transfrom
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = str(Path(self.data_path, self.images[index]))
        image = Image.open(img_path)
        image = self.transfrom(image)
        return image

if __name__ == '__main__':
    data = ImagesData(
        data_path='/media/ywj/File/C-Code/OpenPose-PyTorch/examples/video',
        transfrom=get_transform())
    data_loader = DataLoader(data, batch_size=15, sampler=None)
    for image in tqdm(data_loader, total=len(data_loader)):
        print(image.size())