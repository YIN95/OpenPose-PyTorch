# -*- coding:utf-8 -*-
"""
Author:
    Wenjie Yin, yinw@kth.se
    Reference: https://github.com/Hzzone/pytorch-openpose
"""

import os
import cv2
import math
import time
import copy
import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

from .utils import padRightDownCorner, transfer, findLastIndex, draw_bodypose
from ..networks import BodyModel
from ..dataloader import ImagesData

from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

E = 1e-6


class Body(object):
    def __init__(self, model_path):
        # load model
        self.model = BodyModel()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        model_dict = transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

        # find connection in the specified sequence, center 29 is in the position 15
        self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], 
                        [6, 7], [7, 8], [2, 9], [9, 10],
                        [10, 11], [2, 12], [12, 13], [13, 14], 
                        [2, 1], [1, 15], [15, 17], [1, 16], 
                        [16, 18], [3, 17], [6, 18]]
        # the middle joints heatmap correpondence
        self.mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], 
                       [41, 42], [43, 44], [19, 20], [21, 22],
                       [23, 24], [25, 26], [27, 28], [29, 30], 
                       [47, 48], [49, 50], [53, 54], [51, 52],
                       [55, 56], [37, 38], [45, 46]]

    def removepad_resize(self, input, stride, size, origin_size):
        output = np.transpose(input, (0, 2, 3, 1))
        # input: (1, 19, 23, 41)

        # transpose: (1, 19, 23, 41) -> (1, 23, 41, 19)
        output = np.transpose(input, (0, 2, 3, 1))
        # resize: (23, 41, 19) -> (184, 328, 19)
        # peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]

        # output[i, :] = [cv2.resize(output[i, :], (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC), for i in len(output)]
        output = cv2.resize(output, (0, 0), fx=stride,
                                fy=stride, interpolation=cv2.INTER_CUBIC)
        # remove padding: (184, 328, 19) -> (184, 327, 19)
        output = output[:size[0], :size[1], :]
        # resize: (184, 327, 19) -> (720, 1280, 19)
        output = cv2.resize(
            output, (origin_size[1], origin_size[0]), interpolation=cv2.INTER_CUBIC)

        return output

    def __call__(self, dataset: ImagesData):

        # setting
        dataloader = DataLoader(dataset, batch_size=1, sampler=None)

        search_scale = dataset.search_scale
        boxsize = dataset.boxsize
        stride = dataset.stride
        padValue = dataset.padValue
        origin_size = dataset.origin_size
        thre1 = 0.1
        thre2 = 0.05
        pad = dataset.padding
        size = dataset.size

        # load data
        count = 0
        for data, origin in tqdm(dataloader):
            # heatmap, paf
            if torch.cuda.is_available():
                data = data.cuda()
            with torch.no_grad():
                # PAF, HeatMap
                Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(data)

            # output of the model
            Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().numpy()
            Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().numpy()
            
            heatmap = self.removepad_resize(Mconv7_stage6_L2, stride, size, origin_size)
            paf = self.removepad_resize(Mconv7_stage6_L1, stride, size, origin_size)
     
            all_peaks = []
            peak_counter = 0

            for part in range(18):
                map_ori = heatmap[:, :, part]
                one_heatmap = gaussian_filter(map_ori, sigma=3)

                map_left = np.zeros(one_heatmap.shape)
                map_left[1:, :] = one_heatmap[:-1, :]
                map_right = np.zeros(one_heatmap.shape)
                map_right[:-1, :] = one_heatmap[1:, :]
                map_up = np.zeros(one_heatmap.shape)
                map_up[:, 1:] = one_heatmap[:, :-1]
                map_down = np.zeros(one_heatmap.shape)
                map_down[:, :-1] = one_heatmap[:, 1:]

                peaks_binary = np.logical_and.reduce(
                    (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > thre1))
                peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(
                    peaks_binary)[0]))  # note reverse
                peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
                peak_id = range(peak_counter, peak_counter + len(peaks))
                peaks_with_score_and_id = [
                    peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

                all_peaks.append(peaks_with_score_and_id)
                peak_counter += len(peaks)

            connection_all = []
            special_k = []
            mid_num = 10

            for k in range(len(self.mapIdx)):
                score_mid = paf[:, :, [x - 19 for x in self.mapIdx[k]]]
                candA = all_peaks[self.limbSeq[k][0] - 1]
                candB = all_peaks[self.limbSeq[k][1] - 1]
                nA = len(candA)
                nB = len(candB)
                indexA, indexB = self.limbSeq[k]
                if (nA != 0 and nB != 0):
                    connection_candidate = []
                    for i in range(nA):
                        for j in range(nB):
                            vec = np.subtract(candB[j][:2], candA[i][:2])
                            norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                            vec = (vec + E) / (norm + E)

                            startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                                np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                            vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]
                                              for I in range(len(startend))])
                            vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]
                                              for I in range(len(startend))])

                            score_midpts = np.multiply(
                                vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                            score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                                (0.5 * origin_size[0] + E) / (norm - 1 + E), 0)
                            criterion1 = len(np.nonzero(score_midpts > thre2)[
                                0]) > 0.8 * len(score_midpts)
                            criterion2 = score_with_dist_prior > 0
                            if criterion1 and criterion2:
                                connection_candidate.append(
                                    [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                    connection_candidate = sorted(
                        connection_candidate, key=lambda x: x[2], reverse=True)
                    connection = np.zeros((0, 5))
                    for c in range(len(connection_candidate)):
                        i, j, s = connection_candidate[c][0:3]
                        if (i not in connection[:, 3] and j not in connection[:, 4]):
                            connection = np.vstack(
                                [connection, [candA[i][3], candB[j][3], s, i, j]])
                            if (len(connection) >= min(nA, nB)):
                                break

                    connection_all.append(connection)
                else:
                    special_k.append(k)
                    connection_all.append([])

            # last number in each row is the total parts number of that person
            # the second last number in each row is the score of the overall configuration
            subset = -1 * np.ones((0, 20))
            candidate = np.array(
                [item for sublist in all_peaks for item in sublist])

            for k in range(len(self.mapIdx)):
                if k not in special_k:
                    partAs = connection_all[k][:, 0]
                    partBs = connection_all[k][:, 1]
                    indexA, indexB = np.array(self.limbSeq[k]) - 1

                    for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                        found = 0
                        subset_idx = [-1, -1]
                        for j in range(len(subset)):  # 1:size(subset,1):
                            if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                                subset_idx[found] = j
                                found += 1

                        if found == 1:
                            j = subset_idx[0]
                            if subset[j][indexB] != partBs[i]:
                                subset[j][indexB] = partBs[i]
                                subset[j][-1] += 1
                                subset[j][-2] += candidate[partBs[i].astype(
                                    int), 2] + connection_all[k][i][2]
                        elif found == 2:  # if found 2 and disjoint, merge them
                            j1, j2 = subset_idx
                            membership = ((subset[j1] >= 0).astype(
                                int) + (subset[j2] >= 0).astype(int))[:-2]
                            # merge
                            if len(np.nonzero(membership == 2)[0]) == 0:
                                subset[j1][:-2] += (subset[j2][:-2] + 1)
                                subset[j1][-2:] += subset[j2][-2:]
                                subset[j1][-2] += connection_all[k][i][2]
                                subset = np.delete(subset, j2, 0)
                            else:  # as like found == 1
                                subset[j1][indexB] = partBs[i]
                                subset[j1][-1] += 1
                                subset[j1][-2] += candidate[partBs[i].astype(
                                    int), 2] + connection_all[k][i][2]

                        # if find no partA in the subset, create a new subset
                        elif not found and k < 17:
                            row = -1 * np.ones(20)
                            row[indexA] = partAs[i]
                            row[indexB] = partBs[i]
                            row[-1] = 2
                            row[-2] = sum(candidate[connection_all[k][i,
                                                                      :2].astype(int), 2]) + connection_all[k][i][2]
                            subset = np.vstack([subset, row])
            # delete some rows of subset which has few parts occur
            deleteIdx = []
            for i in range(len(subset)):
                if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                    deleteIdx.append(i)
            subset = np.delete(subset, deleteIdx, axis=0)

            # subset: n*20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
            # candidate: x, y, score, id

            origin_images = origin[0].numpy().transpose(1, 2, 0)*256
            origin_images = cv2.cvtColor(origin_images, cv2.COLOR_BGR2RGB)
            canvas = copy.deepcopy(origin_images)
            canvas, target_points = draw_bodypose(canvas, candidate, subset)
            saveImages_path = '/media/ywj/File/C-Code/OpenPose-PyTorch/examples/skeletonImages'
            temp_path = str(Path(saveImages_path, str(count)+'.jpg'))
            count += 1
            cv2.imwrite(temp_path, canvas.astype(np.uint8))

        # return candidate, subset


def video2skeleton2D(video_images_path, model_path, saveImages=False, saveImages_path=''):
    if saveImages:
        if not os.path.isdir(saveImages_path):
            os.makedirs(saveImages_path)

    body_estimation = Body(model_path)
    data = ImagesData(data_path=video_images_path, mode='default')

    body_estimation(data)

    # for inputs in tqdm(data_loader):
    #     candidate, subset = body_estimation(inputs)
    #     canvas = copy.deepcopy(inputs)
    #     canvas, target_points = draw_bodypose(canvas, candidate, subset)

    #     if saveImages:
    #         temp_path = str(Path(saveImages_path, str(i)+'.jpg'))
    #         cv2.imwrite(temp_path, canvas.astype(np.uint8))
