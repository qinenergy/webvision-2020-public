#!/usr/bin/env python

"""Implementation of training and testing functions for embedding."""

__all__ = ['training', 'load_model', 'forward']
__author__ = 'HK,AK'
__date__ = 'August 2018'

import os.path as ops
import re
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from utils.logging_setup import logger
from utils.utils import Averaging, adjust_lr
from utils.utils import dir_check


def _to_one_hot(y, n_dims, dtype=torch.cuda.FloatTensor):
    scatter_dim = len(y.size())
    y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), n_dims).type(dtype)

    return zeros.scatter(scatter_dim, y_tensor, 1)



def training(config, data, **kwargs):
    """Training pipeline for embedding.

    Args:
        data: iterator within dataset
        epochs: how much training epochs to perform
        n_subact: number of subactions in current complex activity
        mnist: if training with mnist dataset (just to test everything how well
            it works)
    Returns:
        trained pytorch model
    """
    logger.debug('create model')
    torch.manual_seed(config["seed"])

    model = kwargs['model']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']

    create_dataloader = lambda x: \
        torch.utils.data.DataLoader(x, batch_size=config["batch_size"],
                                    shuffle=True, num_workers=config["num_workers"])
    if config["sparse"]:
        dataset = data
        data = create_dataloader(dataset)

    cudnn.benchmark = True

    batch_time = Averaging()
    data_time = Averaging()
    losses = Averaging()

    adjustable_lr = config["lr"]

    logger.debug('epochs: %s', config["epochs"])
    for epoch in range(config["epochs"]):
        model.cuda()
        model.train()

        logger.debug('Epoch # %d' % epoch)
        if config["lr_adj"]:
            if epoch % (50) == 0 and epoch > 0:
                adjustable_lr = adjust_lr(optimizer, adjustable_lr)
                logger.debug('lr: %f' % adjustable_lr)

        end_time = time.time()
        # start_time = time.time();

        print(len(data))

        train_acc_epoch = torch.zeros((1, 1))
        time_epoch = time.time()

        for i, (features, labels) in enumerate(data):
            # print i
            data_time.update(time.time() - end_time)
            features = features.float().cuda(non_blocking=True)

            labels= labels.long().cuda()
            #labels_one_hot = _to_one_hot(labels, config["n_classes"])

            output = model(features)

            max_index = output.max(dim=1)[1]
            train_acc = (max_index == labels).sum()
            train_acc_epoch = train_acc_epoch + train_acc

            loss_values = loss(output, labels)
            losses.update(loss_values.item(), features.size(0))

            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()
            '''
            if i % 5000 == 0 and i:
                logger.debug('Epoch: [{0}][{1}/{2}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(data), batch_time=batch_time,
                    data_time=data_time, loss=losses))
            # print(time.time() - start_time);
            # start_time = time.time();
            '''


        logger.debug('duration: %f' % ( time.time() - time_epoch ) )
        logger.debug('train_err: %f' % ( 1 - ((train_acc_epoch.cpu()).numpy()/(len(data)*config["batch_size"]))) )
        logger.debug('loss: %f' % losses.avg)
        losses.reset()

        if epoch % 1 == 0 and config["save_model"]:
            save_dict = {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
            dir_check(config["model_folder"])
            logger.debug(
                'Saving model to: %s' % ops.join(config["model_folder"], '%s%d.pth.tar' % (config["log_str"], epoch)))
            torch.save(save_dict, ops.join(config["model_folder"], '%s%d.pth.tar' % (config["log_str"], epoch)))
            logger.debug(
                'Saving probs to: %s' % ops.join(config["model_folder"], '%s%d.probs' % (config["log_str"], epoch)))
            data.dataset.save_probs(ops.join(config["model_folder"], '%s%d.probs' % (config["log_str"], epoch)));

        if config["sparse"]:
            dataset.next_epoch()
            data = create_dataloader(dataset)

    if config["save_model"]:
        save_dict = {'epoch': config["epochs"],
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        dir_check(config["model_folder"])
        logger.debug(
            'Saving model to: %s' % ops.join(config["model_folder"], '%s%d.pth.tar' % (config["log_str"], epoch)))
        torch.save(save_dict, ops.join(config["model_folder"], '%s%d.pth.tar' % (config["log_str"], epoch)))
        logger.debug(
            'Saving probs to: %s' % ops.join(config["model_folder"], '%s%d.probs' % (config["log_str"], epoch)))
        data.dataset.save_probs(ops.join(config["model_folder"], '%s%d.probs' % (config["log_str"], epoch)));

    return model


def load_model(config, epoch=None):
    if config["resume_str"]:
        resume_str = config["resume_str"]
        if resume_str.endswith('.pth.tar'):
            search = re.search(r'(.*_)\d*.pth.tar', resume_str)
            resume_str = search.group(1)
    else:
        resume_str = config["log_str"]
    epoch = config["epochs"] if epoch is None else epoch
    logger.debug('Loading model from: %s' % ops.join(config["model_folder"], '%s%d.pth.tar' % (resume_str, epoch)))
    checkpoint = torch.load(ops.join(config["model_folder"], '%s%d.pth.tar' % (resume_str, epoch)))
    checkpoint = checkpoint['state_dict']
    logger.debug('loaded model: ' + '%s%d.pth.tar' % (resume_str, epoch))
    return checkpoint


def forward(config, dataloader, model):
    logger.debug('forward data and save probabilities')
    torch.manual_seed(config["seed"])
    model.eval()
    model.cpu()  # not sure that need it

    with torch.no_grad():
        for idx, (features, labels) in enumerate(dataloader):
            test_feat = np.ones((4, 2048))
            test_feat[0, :] *= 0
            test_feat[1, :] *= 2
            test_feat[2, :] = features[0, :].numpy()
            # features = torch.Tensor(test_feat)
            output = model(features).numpy()
            output_labels = np.argmax(output, axis=1)
            features = features.numpy()
            w = model.fc1.weight.detach().numpy()
            w_last = model.fc_last.weight.detach().numpy()
            bias_last = model.fc_last.bias.detach().numpy()
            bias = model.fc1.bias.detach().numpy()
            # wmean0 = np.mean(w, axis=0)
            wsum = np.sum(w, axis=1)
            print(1)
