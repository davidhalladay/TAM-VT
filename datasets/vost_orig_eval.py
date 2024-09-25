"""
borrowed from `code/VISOR-VOS/eval.py``
"""

from pathlib import Path
from typing import Dict, List

import numpy as np

import util.dist as dist

import json
import warnings
from functools import reduce
from util.box_ops import np_box_iou

import sys
from typing import Any, Dict, List, Sequence, Union
from pprint import pprint
from util.vos_utils import db_eval_iou, db_eval_boundary


def evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
    if all_res_masks.shape[0] > all_gt_masks.shape[0]:
        sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
        sys.exit()
    elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
        zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
        all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
    j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
    # print('J_metric: ', j_metrics_res.shape)
    for ii in range(all_gt_masks.shape[0]):
        if 'J' in metric:
            j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        if 'F' in metric:
            f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
    return j_metrics_res, f_metrics_res


def db_statistics(per_frame_values):
    """ Compute mean,recall and decay from per-frame evaluation.
    Arguments:
        per_frame_values (ndarray): per-frame evaluation

    Returns:
        M,O,D (float,float,float):
            return evaluation statistics: mean,recall,decay.
    """

    # strip off nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        O = np.nanmean(per_frame_values > 0.5)

    N_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [per_frame_values[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])

    return M, O, D


class VOSTOrigEvaluator(object):
    def __init__(self, ):
        self.metric = ['J', 'F']
        self.metrics_res = {}

        self.metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
        self.metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

    def accumulate(self):
        pass

    def update(self, all_gt_masks, all_res_masks):
        j_metrics_res, f_metrics_res = evaluate_semisupervised(
                                all_gt_masks, all_res_masks, None, self.metric)
        # import ipdb; ipdb.set_trace()
        for ii in range(all_gt_masks.shape[0]):
            if 'J' in self.metric:
                [JM, JR, JD] = db_statistics(j_metrics_res[ii])
                self.metrics_res['J']["M"].append(JM)
                self.metrics_res['J']["R"].append(JR)
                self.metrics_res['J']["D"].append(JD)
            if 'F' in self.metric:
                [FM, FR, FD] = db_statistics(f_metrics_res[ii])
                self.metrics_res['F']["M"].append(FM)
                self.metrics_res['F']["R"].append(FR)
                self.metrics_res['F']["D"].append(FD)

    def save(
        self,
        tsa_weights,
        text_weights,
        spatial_weights,
        pred_sted,
        image_ids,
        video_ids,
    ):
        pass

    def postproces_predictions(self, predictions):
        pass

    def synchronize_between_processes(self):
        print("Multi-GPU support for eval is NOT WORKING!!!!!")

    def summarize(self):
        if dist.is_main_process():
            J, F = self.metrics_res['J'], self.metrics_res['F']

            final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
            g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                            np.mean(F["D"])])

            print("scores: [J&F-Mean, J-Mean, J-Recall, J-Decay, F-Mean, F-Recall, F-Decay] are: ", g_res)

        return None, None
