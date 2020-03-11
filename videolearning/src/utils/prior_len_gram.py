#!/usr/bin/env python

"""
"""

__author__ = 'AK'
__date__ = 'December 2018'

import os.path as ops
import re

import numpy as np
from utils.arg_pars import opt


def estimate_prior(video_list):
    #     if ops.exists(ops.join(opt.dataset_root, opt.prior)):
    #         return
    #     prior = np.ones((513, )) * 0.02
    #     with open(ops.join(opt.dataset_root, opt.prior), 'w') as f:
    #         for single_prior in prior:
    #             f.write('%f\n' % single_prior)

    prior = np.zeros((513,))
    video_length = {}
    video_action_sets = {}
    with open(ops.join(opt.dataset_root, 'common_descriptors.txt'), 'r') as f:
        for line in f:
            search = re.search(r'(.*.txt)\s*(\d*)([\s*\d*]*)', line)
            video_name = search.group(1)
            n_frames = search.group(2)
            video_length[video_name] = int(n_frames)
            action_set = search.group(3).strip().split()
            action_set = [int(i) for i in action_set]
            video_action_sets[video_name] = action_set
    for video_name in video_action_sets:
        video_actions = video_action_sets[video_name]
        n_frames = video_length[video_name]
        for action in video_actions:
            prior[action] += n_frames

    prior = prior / np.sum(prior)
    # set bg prior separtly
    prior[-1] = 3.7440000e-02;

    with open(ops.join(opt.dataset_root, opt.prior), 'w') as f:
        for single_prior in prior:
            f.write('%f\n' % single_prior)


def estimate_length_mean(video_list):
    #     if ops.exists(ops.join(opt.dataset_root, opt.length_mean)):
    #         return
    mean_length = np.ones((513,)) * 500
    with open(ops.join(opt.dataset_root, opt.length_mean), 'w') as f:
        for single_mean in mean_length:
            f.write('%f\n' % single_mean)

    return;

    mean_length = np.zeros((513,))
    action_counter = np.zeros((513,))
    video_length = {}
    video_action_sets = {}
    with open(ops.join(opt.dataset_root, 'common_descriptors.txt'), 'r') as f:
        for line in f:
            search = re.search(r'(.*.txt)\s*(\d*)([\s*\d*]*)', line)
            video_name = search.group(1)
            n_frames = search.group(2)
            video_length[video_name] = int(n_frames)
            action_set = search.group(3).strip().split()
            action_set = [int(i) for i in action_set]
            video_action_sets[video_name] = action_set
    for video_name in video_list:
        video_actions = video_action_sets[video_name]
        n_frames = video_length[video_name]
        for action in video_actions:
            mean_length[action] += n_frames
            action_counter[action] += 1

    mean_length /= action_counter
    mean_length = np.nan_to_num(mean_length)
    with open(ops.join(opt.dataset_root, opt.length_mean), 'w') as f:
        for single_mean in mean_length:
            f.write('%f\n' % single_mean)
