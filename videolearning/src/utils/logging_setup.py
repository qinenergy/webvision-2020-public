#!/usr/bin/env python

"""Logger parameters for the entire process.
"""

__author__ = 'AK'
__date__ = 'November 2018'

import datetime
import logging
import os
import re
import sys
from os.path import join

logger = logging.getLogger('basic')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

filename = sys.argv[0]
search = re.search(r'\/*(\w*).py', filename)
filename = search.group(1)


def path_logger(config):
    global logger
    log_dir = join(config["dataset_root"], 'logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    path_logging = join(config["dataset_root"], 'logs', '%s%s(%s)' %
                        (config["log_str"], filename,
                         str(datetime.datetime.now())))
    fh = logging.FileHandler(path_logging, mode='w')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelno)s - %(filename)s - '
                                  '%(funcName)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    for arg in config:
        logger.debug('%s: %s' % (arg, config[arg]))

    return logger
