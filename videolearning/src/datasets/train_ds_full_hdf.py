"""
"""

__author__ = 'AK'
__date__ = 'December 2018'

import time

import h5py
import numpy as np
from torch.utils.data import Dataset
from utils.logging_setup import logger


class DatasetMultic(Dataset):
    config = []
    flag_loading = False;
    idx_loading_frames = 0;
    idx_loading_videos = 0;

    training_sample_count = [];

    def __init__(self, config, video_list, name_parser, label2idx):

        self.config = config

        # count how many frames we got
        temp_labels = []
        for video_idx, video_name in enumerate(video_list):
            #print(video_name)
            hf = h5py.File(video_name, 'r');
            temp_labels_h5 = hf.get('labels')[()];
            hf.close();
            temp_labels_h5 = temp_labels_h5[:, 0]
            temp_labels_h5 = np.squeeze(temp_labels_h5)
            temp_labels.append(temp_labels_h5)

        temp_labels = np.asarray(temp_labels)
        temp_labels = np.concatenate(temp_labels, axis=0)
        self.features = np.zeros((len(temp_labels), self.config["feature_dim"]), dtype=np.float16);
        self.action_set_idxs = np.squeeze(np.zeros((len(temp_labels), 1)));

        self.videolist = video_list;
        self.label2idx = label2idx;
        self.n_classes = len(label2idx)
        self.name_parser = name_parser;

        videos_loaded = 0;

        start_time = time.time();
        for video_idx, video_name in enumerate(video_list):
            if videos_loaded % 5 == 0:
                logger.debug('Processing video : %d' % video_idx)

            file_in = video_name
            hf = h5py.File(file_in, 'r');
            temp_feat = hf.get('data')[()];
            temp_labels = hf.get('labels')[()];
            hf.close();
            temp_labels = temp_labels[:, 0]
            tmp_frames = len(temp_feat);

            self.idx_loading_frames_range = len(temp_labels)

            self.features[self.idx_loading_frames:self.idx_loading_frames + self.idx_loading_frames_range,:] = temp_feat;
            self.action_set_idxs[self.idx_loading_frames:self.idx_loading_frames + self.idx_loading_frames_range] = temp_labels;

            self.idx_loading_frames = self.idx_loading_frames + self.idx_loading_frames_range;

            videos_loaded = videos_loaded + 1;

        np.int64(self.action_set_idxs);
        print('Data loading took ' + str(time.time() - start_time));
        print('----');

        uniq_action_set = np.unique(self.action_set_idxs)

        self.uniq_idx2l = {}
        self.uniq_l2idx = {}
        if len(uniq_action_set) != np.max(uniq_action_set) + 1:
            print('Please check class indices')

        logger.debug('found %d indices' % len(uniq_action_set))
        logger.debug('found %d max index' % (np.max(uniq_action_set) + 1))
        logger.debug('found %d classes' % self.n_classes)

        if self.config["balance_training"] > 0:
            self.features, self.action_set_idxs = self.balance(self.features, self.action_set_idxs)

        self.rand_mapping = np.random.permutation(len(self.action_set_idxs))

        # update sample count
        self.training_sample_count = np.squeeze(np.zeros((np.int32(self.n_classes), 1)))
        for i_idx in range(np.int32(self.n_classes)):
            self.training_sample_count[i_idx] = (self.action_set_idxs == i_idx).sum();

        print('----');

    def loadNewData(self, idx):
        print("No need to load new videos ... ")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[self.rand_mapping[idx]], np.dtype('int64').type(self.action_set_idxs[self.rand_mapping[idx]])

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
        # no class should have more than roughly 10 x mean of the dataset
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
