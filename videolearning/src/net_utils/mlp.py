#!/usr/bin/env python

"""
"""

__author__ = 'AK'
__date__ = 'December 2018'

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.logging_setup import logger


class MLP(nn.Module):

    def __init__(self, config, n_classes):
        super(MLP, self).__init__()
        self.config = config
        self.n_classes = n_classes
        self.fc1 = nn.Linear(config["feature_dim"], config["embed_dim"])
        self.fc_last = nn.Linear(config["embed_dim"], n_classes)
        self._init_weights()

    def forward(self, x):
        if self.config["act_func"] == 'relu':
            x = F.relu(self.fc1(x))
        if self.config["act_func"] == 'sigmoid':
            x = F.sigmoid(self.fc1(x))
        x = self.fc_last(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, self.config["init_mean"], self.config["init_var"])
                nn.init.constant_(m.bias, self.config["bias"])


def create_model(config, n_classes):
    torch.manual_seed(config["seed"])
    model = MLP(config, n_classes).cuda()
    loss = nn.CrossEntropyLoss(reduction='mean').cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"])
    logger.debug(str(model))
    logger.debug(str(loss))
    logger.debug(str(optimizer))
    return model, loss, optimizer
