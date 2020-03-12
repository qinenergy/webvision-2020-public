#!/usr/bin/env python

"""
"""

__author__ = 'HK,AK'
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
from utils.utils import dir_check, join_return_stat, parse_return_stat


def test(config, video_name, label2idx, idx2label, folder_probs, folder_seg):
    print(folder_probs, video_name)
    probs = np.load(ops.join(folder_probs, video_name + '.npy'))
    if np.min(probs) == 0:
        probs[probs == 0] = np.inf
        probs[probs == np.inf] = np.min(probs)
    log_probs = np.log(probs)

    mean_lengths = np.squeeze(np.ones((len(idx2label), 1)) * 150)
    length_model = FlatModel(mean_lengths, max_length=2000)

    file_grammar_path = ops.join(config["transcripts"], video_name + '.txt')
    grammar = PathGrammar(file_grammar_path, label2idx)

    viterbi_decoder = Viterbi(grammar, length_model, frame_sampling=20, max_hypotheses=50000)
    score, labels, segments = viterbi_decoder.decode(log_probs)

    # write result to file
    dir_check(ops.join(folder_seg))
    out_file = ops.join(folder_seg, video_name + '.txt')
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

    # create dir if needed
    dir_check(ops.join(config["out_segmentation"]))

    # sample code for computing segmentation and accuracy for every 2 epochs int the range of 0-50
    all_res = []
    epochs_processed = []
    for i in range(0, 50, 2):

        try:

            config["test_epoch"] = i
            folder_probs = ops.join(config["out_probs"], str(config["test_epoch"]))
            folder_seg = ops.join(config["out_segmentation"], str(config["test_epoch"]))

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

            file_list_gt = []
            if config["gt"] != '':
                for file in glob.glob(ops.join(config["gt"], "*.txt")):
                    file = ops.basename(file)
                    file_list_gt.append(file)

            logger.debug('Found ' + str(len(file_list_gt)) + ' data files in ' + config["gt"] + '... ')

            file_list_probs = []
            if config["out_probs"] != '':
                for file in glob.glob(ops.join(folder_probs, "*.npy")):
                    file = ops.basename(file)
                    file_list_probs.append(file)

            logger.debug('Found ' + str(len(file_list_probs)) + ' data files in ' + folder_probs + '... ')

            mean_iou = []
            for video_name in file_list_gt:
                video_name = ops.splitext(video_name)[0]
                logger.debug(video_name)
                labels, video_gt = test(config, video_name, label2idx, idx2label, folder_probs, folder_seg)

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
            logger.debug("-------------------------------------")
            logger.debug("IoU - Epoch " + str(i) + ": " + str(np.mean(np.asarray(mean_iou))))
            logger.debug("-------------------------------------")

            epochs_processed.append(i)

        except:
            break;

    logger.debug('Epochs processed: ')
    logger.debug(epochs_processed)
    logger.debug('IoU per epoch: ')
    logger.debug(all_res)

    print('DONE!')