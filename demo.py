# -*- coding:utf-8 -*-
"""
Author:
    Wenjie Yin, yinw@kth.se
    Reference: https://github.com/Hzzone/pytorch-openpose
"""

import sys
sys.path.insert(0, 'python')
import openpose as op
import matplotlib.pyplot as plt
import cv2
import copy

body_estimation = op.estimations.Body('openpose/models/body_pose_model.pth')
test_image = 'imgs/d2.png'
oriImg = cv2.imread(test_image)
candidate, subset = body_estimation(oriImg)
canvas = copy.deepcopy(oriImg)
canvas, target_points = op.estimations.draw_bodypose(oriImg, candidate, subset)

plt.imshow(canvas[:, :, [2, 1, 0]])
plt.show()

