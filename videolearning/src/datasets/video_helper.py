#!/usr/bin/env python

"""
"""

__author__ = 'AK'
__date__ = 'December 2018'

import numpy as np


class Video:
    '''Helper class'''

    def __init__(self, start, n_frames, action_set, name):
        self.start = start
        self.n_frames = n_frames
        self.action_set = action_set
        self.mask = None
        self.name = name
        self.path = ''
        self.rest_frames = set(np.arange(n_frames))

    def processed_frames(self, frame_range):
        self.rest_frames = self.rest_frames - set(frame_range)

    def update_range(self, total):
        self.mask = 0;  # np.zeros((total, 1), dtype=bool)
        # self.mask[self.start: self.start + self.n_frames] = True
