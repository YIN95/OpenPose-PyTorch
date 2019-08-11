# -*- coding:utf-8 -*-
"""
Author:
    Wenjie Yin, yinw@kth.se
    Reference: https://github.com/Hzzone/pytorch-openpose
"""

import PIL.Image as Image
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
import torch
import os
import cv2
from torch.utils.data import Dataset, DataLoader

def get_transform(mode='default', size=(0, 0), stride=8, padding=(0, 0, 0, 0), padValue=128):
    assert (size != (0, 0)), \
        "The size of the image should not be (0, 0)."

    if mode in ['default']:

        return transforms.Compose([
            transforms.Resize(size=size, interpolation=Image.BICUBIC),
            transforms.Pad(padding=padding, fill=padValue, padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])


class ImagesData(Dataset):
    def __init__(self, data_path, mode='default'):
        super(ImagesData, self).__init__()

        self.data_path = data_path
        self.images = os.listdir(data_path)

        self.mode = mode
        # (1280, 720)
        self.origin_size = Image.open(
            str(Path(self.data_path, self.images[0]))).size
        # (720, 1280)
        self.origin_size = (self.origin_size[1], self.origin_size[0])
        self.search_scale = 0.5
        self.boxsize = 368
        self.size = (
            int(self.search_scale * self.boxsize),
            int(self.search_scale * self.boxsize * self.origin_size[1] / self.origin_size[0]))

        self.stride = 8
        self.padValue = 128

        self.padding = 4 * [0]
        # right
        if self.size[1] % self.stride != 0:
            self.padding[2] = self.stride - (self.size[1] % self.stride)
        # down
        if self.size[0] % self.stride != 0:
            self.padding[3] = self.stride - (self.size[0] % self.stride)
        self.padding =tuple(self.padding)

        self.transfrom = get_transform(
            mode=self.mode,
            size=self.size,
            stride=self.stride,
            padding=self.padding,
            padValue=self.padValue)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = str(Path(self.data_path, self.images[index]))
        image = Image.open(img_path)
        image = self.transfrom(image)
        # print(image.size())
        return image


if __name__ == '__main__':
    data = ImagesData(
        data_path='/media/ywj/File/C-Code/OpenPose-PyTorch/examples/video',
        mode='default')
    data_loader = DataLoader(data, batch_size=1, sampler=None)
    for image in tqdm(data_loader, total=len(data_loader)):
        print(image.size())
