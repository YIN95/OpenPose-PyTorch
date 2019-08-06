# -*- coding:utf-8 -*-
"""
Author:
    Wenjie Yin, yinw@kth.se
    Reference: https://github.com/Hzzone/pytorch-openpose
"""

import sys
sys.path.insert(0, 'python')

import cv2
import copy

import openpose as op
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from collections import defaultdict
from pathlib import Path

body_estimation = op.estimations.Body('openpose/models/body_pose_model.pth')

video_root = './imgs'
video_name = 'demo_movie.avi'
cap = cv2.VideoCapture(str(Path(video_root, video_name)))

n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video = cv2.VideoWriter('video.avi',
                        cv2.VideoWriter_fourcc(*'DIVX'),
                        fps,
                        (width, height))
recoder = defaultdict(list)
recoder_keys = ['right_ankle_x', 'right_ankle_y',
                'left_ankle_x', 'left_ankle_y',
                'right_groin_x', 'right_groin_y',
                'left_groin_x', 'left_groin_y']

i_frame = 0
cnt = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    cnt = cnt + 1
    print(cnt)
    if not ret:
        break

    candidate, subset = body_estimation(frame)
    canvas = copy.deepcopy(frame)
    canvas, target_points = op.estimations.draw_bodypose(canvas, candidate, subset)
    for i, t in enumerate(target_points):
        i_x = i*2
        i_y = i*2 + 1
        recoder[recoder_keys[i_x]].append(t[0])
        recoder[recoder_keys[i_y]].append(t[1])
    
    video.write(canvas.astype(np.uint8))


cap.release()
video.release()
# pd.DataFrame(recoder).to_csv('results/positions.csv', index=False)
