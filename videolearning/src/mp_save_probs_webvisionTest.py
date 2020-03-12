#!/usr/bin/env python3

"""
"""

__author__ = 'HK,AK'
__date__ = 'December 2018'

import json
import multiprocessing as mp
import os.path as ops

import numpy as np
import torch
from datasets.test_ds_hdf import TestDataset
from net_utils import mlp
from net_utils.training_network import load_model
from utils.logging_setup import logger
from utils.utils import dir_check, logger_setup


class SaveProbsMP:
    def __init__(self, config, model, dataset, folder_probs):
        self.config = config
        self.folder_probs = folder_probs
        ctx = mp.get_context('spawn')
        self._queue = ctx.Queue()
        torch.manual_seed(self.config["seed"])
        self.model = model
        self.model.cpu()
        self.model.eval()
        self.dataset = dataset
        dir_check(ops.join(self.config["dataset_root"], config["out_probs"]))

        for i in range(len(self.dataset)):
            self._queue.put(i)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        e_x =  e_x / e_x.sum()
        return e_x

    def save_probs(self, n_video):
        features, name = self.dataset[n_video]
        with torch.no_grad():
            output = self.model(features).numpy()

        # compute softmax for linear output ... remove this if neede
        output = self.softmax(output)

        if self.config["get_cond_probs"] > 0:
            # avoid divide by zero ...
            if np.min(output) == 0:
                output[output == 0] = np.inf
                output[output == np.inf] = np.min(output)
            # go to log space ....
            log_probs = np.log(output)

            # prior = np.loadtxt(ops.join(opt.dataset_root, opt.prior))
            prior = np.load(ops.join(self.config["model_folder"],
                                     '%s%d.probs.npy' % (self.config["log_str"], self.config["test_epoch"])));
            prior = np.squeeze(prior);

            log_prior = np.log(prior)
            # log_prior = np.nan_to_num(log_prior)
            log_prior[prior == 0] = 0
            log_probs = log_probs - log_prior
            # set bg separtly
            log_probs[:, -1] = np.mean(log_probs[:, :-1]);

            if np.max(log_probs) > 0:
                log_probs -= 2 * np.max(log_probs)

            output = np.exp(log_probs)

        dir_check(folder_probs)
        np.save(ops.join(folder_probs, name), output)

    def save_probs_queue(self):
        while not self._queue.empty():
            #try:
            n_video = self._queue.get(timeout=3)
            self.save_probs(n_video)
            #except Queue.Empty:
            #    pass

    def save_probs_mp(self, n_threads=1):
        logger.debug('.')
        procs = []
        for i in range(n_threads):
            p = mp.Process(target=self.save_probs_queue)
            procs.append(p)
            p.start()
        for p in procs:
            p.join()


def save_mp(config, folder_probs):
    logger.debug('Multiprocessing: forward data and save probabilities')
    dataset = TestDataset(config)
    model, loss, optimizer = mlp.create_model(config, n_classes=513)
    model.load_state_dict(load_model(config, epoch=config["test_epoch"]))
    save_item = SaveProbsMP(config, model, dataset, folder_probs)
    save_item.save_probs_mp()


def save_probs(config, dataloader, model):
    dir_check(ops.join(config["dataset_root"], config["out_probs"]))

    logger.debug('forward data and save probabilities')
    torch.manual_seed(config["seed"])
    model.eval()
    model.cpu()  # not sure that need it

    with torch.no_grad():
        for idx, (features, name) in enumerate(dataloader):
            output = model(features).numpy()
            np.save(ops.join(config["dataset_root"], config["out_probs"], name), output)


def save(config):
    model, ctloss, optimizer = mlp.create_model(n_classes=513)
    model.load_state_dict(load_model(epoch=config["test_epoch"]))
    dataset = TestDataset(config)
    dataloader = iter(dataset)
    save_probs(config, dataloader, model)


if __name__ == '__main__':

    # load config file
    with open('../config/train_config_relu_org_files_2048dim.json') as config_file:
        config = json.load(config_file)
    print(config["out_probs"])
    logger_setup(config)

    # create dir if needed
    dir_check(ops.join(config["out_probs"]))

    # sample code for computing output probabilities for every two epochs in the range of 0-50
    len_last_str = 0
    epochs_processed = []
    for i in range(0, 50, 2):
        config["test_epoch"] = i
        folder_probs= ops.join(config["out_probs"], str(config["test_epoch"]))
        len_last_str = len(str(i))

        logger_setup(config)

        try:
            save_mp(config, folder_probs)
            epochs_processed.append(i)
        except:
            break;

    logger.debug('Epochs processed: ')
    logger.debug(epochs_processed)
    print('Done!')