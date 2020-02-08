#!/usr/bin/env python

"""
"""

__author__ = 'AK'
__date__ = 'December 2018'

import os
import os.path as ops
import time

import h5py
import numpy as np
from datasets.video_helper import Video
from torch.utils.data import Dataset
from utils.logging_setup import logger
from utils.utils import dir_check
from utils.utils import join_data


class DatasetMulticSparse(Dataset):
    config = []

    flag_loading = False;
    idx_loading_frames = 0;
    idx_loading_videos = 0;

    training_sample_count = [];
    n_classes = 0;

    max_num_vids = 0

    def __init__(self, config, video_list, name_parser, label2idx):

        self.config = config
        self.max_num_vids = len(video_list)

        # malloc for [num_frames * num_vids , feature_dim] and labels
        self.features = np.zeros((self.config["sparse_num_frames"] * self.max_num_vids, self.config["feature_dim"]));
        self.action_set_idxs = np.squeeze(np.zeros((self.config["sparse_num_frames"] * self.max_num_vids, 1)));

        self.videolist = video_list;
        self.label2idx = label2idx;
        self.n_classes = len(label2idx)
        self.name_parser = name_parser;
        self.videos = [];

        videos_loaded = 0;

        start_time = time.time();
        for video_idx, video_name in enumerate(video_list):

            if videos_loaded % 5 == 0:
                logger.debug('Processing video : %d' % video_idx)

            if videos_loaded >= self.max_num_vids:
                break;

            file_in = video_name
            hf = h5py.File(file_in, 'r');
            temp_feat = hf.get('data')[()];
            temp_labels = hf.get('labels')[()];
            hf.close();
            temp_labels = temp_labels[:, 0]
            tmp_frames = len(temp_feat);

            # load 10000 frames from 1000 videos,  10 frames from each video             
            feat_idx = np.random.randint(tmp_frames, size=self.config["sparse_num_frames"]);
            temp_feat = temp_feat[feat_idx, :]
            assert temp_feat.shape[1] == self.config["feature_dim"]

            temp_action_set_idx = temp_labels[feat_idx];
            temp_action_set_idx = np.squeeze(temp_labels[feat_idx]);
            #
            #             self.features[:] = join_data(self.features, temp_feat, np.vstack)
            #             self.action_set_idxs = np.concatenate((self.action_set_idxs, temp_action_set_idx), axis=None);

            self.features[self.idx_loading_frames:self.idx_loading_frames + self.config["sparse_num_frames"],
            :] = temp_feat;
            self.action_set_idxs[
            self.idx_loading_frames:self.idx_loading_frames + self.config["sparse_num_frames"]] = temp_action_set_idx;

            self.idx_loading_frames = self.idx_loading_frames + self.config["sparse_num_frames"];

            # we keep track of what features have been used so far :
            start = 0 if self.features is None else len(self.features)
            video = Video(start=start,
                          n_frames=tmp_frames,
                          action_set=temp_labels,
                          name=video_name)
            video.processed_frames(feat_idx)
            self.videos.append(video)
            videos_loaded = videos_loaded + 1;

        np.int64(self.action_set_idxs);
        print(time.time() - start_time);
        print('----');

        logger.debug('found indices for %d classes' % len(np.unique(self.action_set_idxs)))

        if self.config["balance_training"] > 0:
            self.features, self.action_set_idxs = self.balance(self.features, self.action_set_idxs)

        self.training_sample_count = np.zeros(self.n_classes);
        # update sample count
        for i_idx in range(self.n_classes):
            self.training_sample_count[i_idx] = self.training_sample_count[i_idx] + (
                        self.action_set_idxs == i_idx).sum();

        print('----');

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        #         action_set_idx = self.action_set_idxs[idx]
        #         label = np.random.choice(self.action_sets[action_set_idx], 1)[0]
        #         return self.features[idx], label

        label = self.action_set_idxs[idx];
        if label > self.n_classes:
            print('Whats wrong')
        return self.features[idx], label

    def _tmp_save(self, idx):
        dir_check(ops.join(self.config["dataset_root"], self.tmp))
        np.save(ops.join(self.config["dataset_root"], self.tmp, '%d.npy' % idx), self.features)
        del self.features
        self.features = None

    def _tmp_read(self):
        del self.features
        self.features = None
        tmp_path = ops.join(self.config["dataset_root"], self.tmp)
        tmp_list = [int(i.split('.')[0]) for i in os.listdir(tmp_path)]
        for file_idx in sorted(tmp_list):
            logger.debug(file_idx)
            tmp_file_path = ops.join(tmp_path, '%d.npy' % file_idx)
            tmp_feat = np.load(tmp_file_path)
            self.features = join_data(self.features, tmp_feat, np.vstack)
            os.remove(tmp_file_path)
        # os.rmdir(tmp_path)

    def next_epoch(self):
        print("loading new videos ... ")
        # time.sleep(5)

        self.idx_loading_frames = 0;
        self.idx_loading_videos = 0;

        videos_loaded = 0;

        start_time = time.time();
        for video_idx, video in enumerate(self.videos):
            if videos_loaded % 5 == 0:
                logger.debug('Processing video : %d' % video_idx)

            if videos_loaded >= self.max_num_vids:
                break;

            file_in = video.name;
            hf = h5py.File(file_in, 'r');
            temp_feat = hf.get('data')[()];
            temp_labels = hf.get('labels')[()];
            hf.close();
            temp_labels = temp_labels[:, 0]
            tmp_frames = len(temp_feat);

            # load frames from rest frames in the video, reset if neccessary
            if len(video.rest_frames) < self.config["sparse_num_frames"]:
                video.rest_frames = set(np.arange(video.n_frames))
            try:
                feat_idx = np.random.choice(list(video.rest_frames),
                                            size=self.config["sparse_num_frames"],
                                            replace=False)
            except ValueError:
                feat_idx = np.random.choice(list(video.rest_frames),
                                            size=self.config["sparse_num_frames"],
                                            replace=True)
            # feat_idx = np.random.randint(tmp_frames, size=self.idx_loading_frames_range);
            temp_feat = temp_feat[feat_idx, :]
            assert temp_feat.shape[1] == self.config["feature_dim"]

            temp_action_set_idx = np.squeeze(temp_labels[feat_idx]);

            self.features[self.idx_loading_frames:self.idx_loading_frames + self.config["sparse_num_frames"],
            :] = temp_feat;
            self.action_set_idxs[
            self.idx_loading_frames:self.idx_loading_frames + self.config["sparse_num_frames"]] = temp_action_set_idx;

            self.idx_loading_frames = self.idx_loading_frames + self.config["sparse_num_frames"];

            # update rest_frames
            video.processed_frames(feat_idx)
            videos_loaded = videos_loaded + 1;

        np.int64(self.action_set_idxs);
        print(time.time() - start_time);
        print('----');

        if self.config["balance_training"] > 0:
            self.features, self.action_set_idxs = self.balance(self.features, self.action_set_idxs)

        # update sample count
        for i_idx in range(self.n_classes):
            self.training_sample_count[i_idx] = self.training_sample_count[i_idx] + (
                        self.action_set_idxs == i_idx).sum();
        print('----');

    def save_probs(self, filename):
        print('-- saving probs --');
        print(filename);

        sum_all = self.training_sample_count.sum();
        tmp_probs = np.zeros((len(self.training_sample_count), 1))
        for i_idx in range(len(self.training_sample_count)):
            tmp_probs[i_idx] = self.training_sample_count[i_idx] / sum_all;

        np.save(filename, tmp_probs)


    def balance(self, feature, index):
        print('-- balancing data --');
        # remove samples from large classes:
        mean_samples = len(index)/self.n_classes;
        # no class should have more than roughly 2 x mean of the dataset
        max_samples = mean_samples * 2;
        # no class should have les than mean of the maximum
        min_samples = mean_samples;

        # first down sample
        idx_remove = []
        for i_idx in range(np.int32(self.n_classes)):
            is_feat = (index == i_idx)
            if is_feat.sum() > max_samples:
                idx_tmp = np.where(is_feat == 1)
                idx_tmp = idx_tmp[0];
                idx_remove.append(np.random.choice(idx_tmp, size=np.int(len(idx_tmp) - max_samples), replace=False))

        if len(idx_remove) > 0:
            idx_remove = np.asarray(idx_remove)
            idx_remove = np.concatenate(idx_remove, axis=0)

            feature = np.delete(feature, idx_remove, 0)
            index = np.delete(index, idx_remove, 0)
            print('removed entries: ' +str(len(idx_remove)));

        # than up sample
        idx_add = []
        for i_idx in range(np.int32(self.n_classes)):
            is_feat = (index == i_idx)
            if is_feat.sum() < min_samples:
                idx_tmp = np.where(is_feat == 1)
                idx_tmp = idx_tmp[0];
                if len(idx_tmp) == 0:
                    continue;
                idx_add.append(np.random.choice(idx_tmp, size=np.int(min_samples - len(idx_tmp)), replace=True))

        if len(idx_add) > 0 :
            idx_add = np.asarray(idx_add)
            idx_add = np.concatenate(idx_add, axis=0)

            feature = np.vstack((feature, feature[idx_add, :]))
            index = np.hstack((index, index[idx_add]))

            print('added entries: ' +str(len(idx_add)));

        return feature, index
