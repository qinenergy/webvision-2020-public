#!/usr/bin/env python

"""
"""

__author__ = 'AK'
__date__ = 'December 2018'

import os.path as ops
import glob
import numpy as np
import json

from utils.logging_setup import logger
from utils.utils import logger_setup
from viterbi_utils.grammar import PathGrammar
from viterbi_utils.length_model import FlatModel
from viterbi_utils.viterbi import Viterbi
from datasets.test_ds_new import TestDataset
from utils.utils import dir_check, join_return_stat, parse_return_stat


def test(config, video_name, label2idx, idx2label):
    probs = np.load(ops.join(config["out_probs"], video_name + '.npy'))
    if np.min(probs) == 0:
        probs[probs == 0] = np.inf
        probs[probs == np.inf] = np.min(probs)
    log_probs = np.log(probs)

    mean_lengths = np.squeeze(np.ones((len(idx2label), 1)) * 150)
    length_model = FlatModel(mean_lengths, max_length=2000)

    file_grammar_path = ops.join(config["dataset_root"], config["transcripts"], video_name + '.txt')
    grammar = PathGrammar(file_grammar_path, label2idx)

    viterbi_decoder = Viterbi(grammar, length_model, frame_sampling=20, max_hypotheses=50000)
    score, labels, segments = viterbi_decoder.decode(log_probs)

    # write result to file
    dir_check(ops.join(config["out_segmentation"]))
    out_file = ops.join(config["out_segmentation"], video_name + '.txt')
    with open(out_file, 'w') as f:
        for label in labels:
            f.write('%s\n' % idx2label[label])

    # read gt
    video_gt = []
    with open(ops.join( config["gt"], video_name + '.txt'), 'r') as f:
        for line in f:
            line = line.strip()
            idx = label2idx[line]
            video_gt.append(idx)

    # set labels and gt to same length
    if len(labels) < len(video_gt):
        # do padding with last label:
        labels_new = np.squeeze(np.zeros((len(video_gt), 1)))
        labels_new[:len(labels)] = labels
        labels_new[len(labels):len(video_gt)] = labels[-1]
        labels = labels_new
    if len(labels) > len(video_gt):
        labels = labels[:len(video_gt)]

    return labels, video_gt


if __name__ == '__main__':
    with open('../config/train_config_relu_org_files_2048dim.json') as config_file:
        config = json.load(config_file)

    all_res = []
    all_res2 = []
    len_last_str = 2
    for i in range(0, 50, 2):
        config["test_epoch"] = i
        config["out_probs"]= config["out_probs"][:-len_last_str]+"_" + str(i)
        config["out_segmentation"] = config["out_segmentation"][:-len_last_str]+"_" + str(i)
        print(config["out_probs"])
        print(config["out_segmentation"])
        len_last_str = len("_" + str(i))

        logger_setup(config)
        np.random.seed(config["seed"])
        # read label2index mapping and index2label mapping
        label2idx = dict()
        idx2label = dict()
        return_stat_all = None
        with open(ops.join(config["label_idx_file"]), 'r') as f:
            content = f.read().split('\n')[0:-1]
            for line in content:
                if len(line) > 2:
                    label2idx[line.split()[1]] = int(line.split()[0])
                    idx2label[int(line.split()[0])] = line.split()[1]

        file_list = []
        if config["out_probs"] != '':
            for file in glob.glob(ops.join(config["out_probs"], "*.npy")):
                file_list.append(file)

        logger.debug('Found ' + str(len(file_list)) + ' data files in ' + config["out_probs"] + '... ')

        dataset = TestDataset(config)
        labels_all = []
        video_gt_all = []
        mean_iou = []
        for video_idx, video_name in enumerate(dataset.names):
            logger.debug(video_idx)
            video_name = ops.splitext(video_name)[0]
            labels, video_gt = test(config, video_name, label2idx, idx2label)
            labels_all.append(labels)
            video_gt_all.append(video_gt)

            # compute framewise IoU for all classes in this video
            unique_classes = np.unique(video_gt)
            IoU_per_classes = np.zeros((len(unique_classes), 1))
            count = 0
            for idx_tmp in unique_classes:
                intersection_tmp = np.sum((np.logical_or((labels == idx_tmp), (video_gt == idx_tmp))).astype(float))
                union_tmp = np.sum((np.logical_and((labels == idx_tmp), (video_gt == idx_tmp))).astype(float))
                if union_tmp == 0 and intersection_tmp == 0:
                    print('Invalid index: ' + str(idx_tmp))
                if union_tmp == 0:
                    IoU_per_classes[count] = 0
                else:
                    IoU_per_classes[count] = union_tmp / intersection_tmp
                count = count + 1

            logger.debug("IoU : " + str(np.mean(IoU_per_classes)))
            mean_iou.append(np.mean(IoU_per_classes))

        all_res.append(np.mean( np.asarray(mean_iou) ))
        logger.debug("IoU over all videos: " + str(np.mean( np.asarray(mean_iou) )))

    print(all_res)

    print('DONE!')
