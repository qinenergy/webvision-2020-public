#!/usr/bin/python3

"""
"""

__author__ = 'HK,AK'
__date__ = 'December 2018'

import glob
import json
import os

import numpy as np
import torch
from datasets.train_ds_full_hdf import DatasetMultic
from datasets.train_ds_sparse_hdf import DatasetMulticSparse
from net_utils import mlp
from net_utils.training_network import training, load_model, forward
from utils.logging_setup import logger
from utils.utils import name_parser, logger_setup

################################################################################
### MAIN                                                                     ###
################################################################################
if __name__ == '__main__':

    with open('../config/train_config_relu_org_files_2048dim.json') as config_file:
        config = json.load(config_file)
    logger_setup(config)
    np.random.seed(config["seed"])

    # read label2index mapping and index2label mapping
    label2idx = dict()
    idx2label = dict()
    with open(os.path.join(config["label_idx_file"]), 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            if len(line) > 2:
                label2idx[line.split()[1]] = int(line.split()[0])
                idx2label[int(line.split()[0])] = line.split()[1]

    n_classes = len(idx2label)
    logger.debug('Found ' + str(n_classes) + ' classes/labels ... ')

    file_list = []
    if config["data_folder"] != '':
        for file in glob.glob(os.path.join(config["data_folder"], "*.h5")):
            file_list.append(file)

    logger.debug('Found ' + str(len(file_list)) + ' data files in ' + config["data_folder"] + '... ')
    if len(file_list) == 0:
        logger.debug('No training files found, please check data folder: ' + config["data_folder"] + '... ')

    model, loss, optimizer = mlp.create_model(config, n_classes)

    if config["sparse"]:
        dataset = DatasetMulticSparse(config, file_list, name_parser, label2idx)
        model = training(config, dataset, model=model, loss=loss, optimizer=optimizer,
                         sparse=True)

    else:
        dataset = DatasetMultic(config, file_list, name_parser, label2idx)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=config["batch_size"],
                                                 num_workers=config["num_workers"])
        model = training(config, dataloader, model=model, loss=loss, optimizer=optimizer)

    print('DONE!')