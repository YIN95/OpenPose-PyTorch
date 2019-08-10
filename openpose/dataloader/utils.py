# -*- coding:utf-8 -*-
"""
Author:
    Wenjie Yin, yinw@kth.se
"""

import cv2
import os

from pathlib import Path
from tqdm import tqdm

def findLastIndex(s, ch):
    temp = [i for i, ltr in enumerate(s) if ltr == ch]
    return temp[-1]

def extractVideoName(path):
    basename = os.path.basename(path)
    index = findLastIndex(basename, '.')
    return basename[:index]

def getSavePath(save_path, file_name):
    return str(Path(save_path, file_name))

def video2images(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    videoName = extractVideoName(video_path)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    
    pbar = tqdm(total=n_frame)
    pbar.set_description("Converting")

    save_path = str(Path(save_path, videoName))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        imageName = "%d.jpg" % count
        cv2.imwrite(str(Path(save_path, imageName)), frame)
        count += 1
        pbar.update()
    pbar.close()
    cap.release()

def funcname(self, parameter_list):
    pass

if __name__ == "__main__":
    video_path = '/media/ywj/File/C-Code/OpenPose-PyTorch/examples/video.avi'
    save_path = '/media/ywj/File/C-Code/OpenPose-PyTorch/results/'

    video2images(video_path, save_path)
    # print(extractVideoName(video_path))