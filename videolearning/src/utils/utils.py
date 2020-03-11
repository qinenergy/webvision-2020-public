#!/usr/bin/env python

"""Utilities for the project"""

__all__ = ['join_data', 'Averaging', 'adjust_lr', 'timing', 'dir_check',
           'parse', 'count_classes', 'AverageLength', 'merge', 'update_opt_str',
           'join_return_stat', 'parse_return_stat', 'name_parser']
__author__ = 'AK'
__date__ = 'August 2018'

import os
import re
import time
from collections import defaultdict

import numpy as np
from utils.logging_setup import logger, path_logger


class Averaging(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def join_data(data1, data2, f):
    """Simple use of numpy functions vstack and hstack even if data not a tuple

    Args:
        data1 (arr): array or None to be in front of
        data2 (arr): tuple of arrays to join to data1
        f: vstack or hstack from numpy

    Returns:
        Joined data with provided method.
    """
    if isinstance(data2, tuple):
        data2 = f(data2)
    if data1 is not None:
        data2 = f((data1, data2))
    return data2


def adjust_lr(optimizer, lr):
    """Decrease learning rate by 0.1 during training"""
    lr = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def timing(f):
    """Wrapper for functions to measure time"""

    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        logger.debug('%s took %0.3f ms ~ %0.3f min ~ %0.3f sec'
                     % (f, (time2 - time1) * 1000.0,
                        (time2 - time1) / 60.0,
                        (time2 - time1)))
        return ret

    return wrap


def dir_check(path):
    """If folder given path does not exist it is created"""
    if not os.path.exists(path):
        os.mkdir(path)


def parse(path):
    """Parsing of logger file

    This function parses log file which was created by logger in this project
    to extract frames and MoF.

    Args:
        path: location of log file on your computer

    Returns:
        seq of MoF
        seq of number of frames
        max # of frames during entire training
    """
    counter = 0
    val = []
    frames = []
    with open(path, 'r') as f:
        for line in f:
            if 'Iteration' in line:
                line = line.split()[-1]
                counter = int(line)
            if 'MoF' in line:
                if 'old' in line:
                    continue
                print(counter, line)
                line = line.split()[-1]
                val.append(float(line))
            if 'frames' in line:
                line = line.split()[-5]
                frames.append(int(line))
        for v in val:
            print(v)
        for fr in frames:
            print(fr)
    print(len(val), np.max(frames), np.where(np.asarray(frames) == np.max(frames)))


def count_classes(path, end):
    """Counting different complex activities in dataset

    Args:
        path: root folder of dataset
        end: file extension of features/videos
    """
    counter = defaultdict(int)
    for f in os.listdir(path):
        if f.endswith(end):
            # print(f)
            f_name = f.split('_')[-1]
            counter[f_name] += 1
    print(counter)


class AverageLength(object):
    """Class helper for calculating average length during segmentation"""

    def __init__(self):
        self._total_length = 0
        self._nmb_of_sgmts = 0

    def add_segments(self, new_segments):
        if new_segments is None:
            return
        assert len(new_segments) > 1
        for idx, start in enumerate(new_segments[:-1]):
            end = new_segments[idx + 1]
            length = end - start
            self._total_length += length
            self._nmb_of_sgmts += 1

    def __call__(self, *args, **kwargs):
        return int(self._total_length / self._nmb_of_sgmts)


def merge(arr1, arr2):
    """Merge two sorted arrays without duplicates

    Args:
        arr1: first sorted array
        arr2: second sorted array
    Returns:
        sorted array comprises items from both input arrays
    """
    i, j = 0, 0
    total = []
    try:
        total_len = len(arr1) + len(arr2)
    except TypeError:
        # if one of the arrays is None
        return arr1 if arr2 is None else arr2
    while i < len(arr1) or j < len(arr2):
        try:
            # i += total[-1] == arr1[i]
            # j += total[-1] == arr2[j]
            comparator = arr1[i] < arr2[j]
            val = arr1[i] if comparator else arr2[j]
            total.append(val)
            i += comparator
            j += not comparator
            if total[-1] == arr1[i]:
                i += 1
                total_len -= 1
                continue
            if total[-1] == arr2[j]:
                j += 1
                total_len -= 1
                continue
        except IndexError:
            arr1 = arr1 if i < len(arr1) else [arr2[-1] + 1]
            arr2 = arr2 if j < len(arr2) else [arr1[-1] + 1]
            i = i if len(arr1) > 1 else 0
            j = j if len(arr2) > 1 else 0
            if total_len == len(total):
                break
    return total


def update_opt_str(config):
    logs_args_map = {'epochs': 'ep',
                     'embed_dim': 'dim',
                     'lr': 'lr',
                     'act_func': '',
                     'init_mean': 'im',
                     'init_var': 'iv',
                     'bias': 'bias'}

    log_str = ''
    logs_args = ['prefix'] + sorted(logs_args_map)
    logs_args_map['prefix'] = ''
    logs_args_map['subaction'] = ''
    for arg in logs_args:
        attr = config[arg]
        arg = logs_args_map[arg]
        if isinstance(attr, bool):
            if attr:
                attr = arg
            else:
                attr = '!' + arg
        else:
            attr = '%s%s' % (arg, str(attr))
        log_str += '%s_' % attr

    config["log_str"] = log_str

    for arg in config:
        logger.debug('%s: %s' % (arg, config[arg]))


def join_return_stat(stat1, stat2):
    keys = ['mof', 'iou']
    stat = {}
    if stat1 is None:
        return stat2
    for key in keys:
        v11, v21 = stat1[key]
        v12, v22 = stat2[key]
        stat[key] = [v11 + v12, v21 + v22]
    return stat


def parse_return_stat(stat):
    keys = ['mof', 'iou']
    for key in keys:
        v1, v2 = stat[key]
        val = v1 / v2
        logger.debug('%s: %f' % (key, val))


def name_parser(videoname, label2index):
    search = re.search(r'\w*_clip\d+_([\w,-]*)', videoname)
    class_labels = search.group(1).split('-')
    class_indexes = [label2index[i] for i in class_labels]
    return class_indexes


def logger_setup(config):
    update_opt_str(config)
    path_logger(config)


if __name__ == '__main__':
    a = [1, 5, 10, 12, 15]
    b = [3, 4, 7, 10]
    merge(a, b)
