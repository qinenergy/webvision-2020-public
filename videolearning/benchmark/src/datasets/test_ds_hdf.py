#!/usr/bin/env python

"""
"""

__author__ = 'AK'
__date__ = 'December 2018'

import os
import os.path as ops

import numpy as np
import torch
from utils.logging_setup import logger


class TestDataset:
    def __init__(self, config):

        self.config = config

        self.features = None
        self.pathes = []
        self.names = []
        self.counter = 0
        for idx, filename in enumerate(os.listdir(ops.join(self.config["dataset_root"], self.config["test_feat"]))):
            path = ops.join(self.config["dataset_root"], self.config["test_feat"], filename)
            self.pathes.append(path)
            self.names.append(filename)

        logger.debug(' %d videos' % len(self.names))

    def __getitem__(self, idx):
        if idx % 10 == 0:
            logger.debug('processed [%d]/[%d] videos' % (idx, len(self)))
        features = np.load(self.pathes[idx])
        return torch.Tensor(features), self.names[idx]

    def __len__(self):
        return len(self.pathes)

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < len(self):
            tensor_name = self.__getitem__(self.counter)
            self.counter += 1
            return tensor_name
        else:
            raise StopIteration
