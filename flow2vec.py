from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from configparser import ConfigParser
import generator
import logging
import math
import os
import pathlib
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

config_file = "./sample_config.ini"
cp = ConfigParser()
cp.read(config_file)

# Tensorboard log directory
log_dir = cp["DEFAULT"].get("log_dir")

def logger():
    """
    
    """
    logger = logging.getLogger('flow2vec')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    return logger

logs = logger()

def make_log_dir(log_dir: str):
    """Create tensorboard log directory
    
    Arguments:
        log_dir {str} -- directory on file system to save Tensorboard Log File(s)
    """

    log_dir = pathlib.Path(log_dir)

    if log_dir.exists():
        logs.info(f'Tensoboard log directory {log_dir} already exists!')
    else:
        os.makedirs(log_dir)
        logs.info(f'Tensorboard Log Directory {log_dir} created!')

def build_dataset(flows: int, n_flows:int):
    """Process dataframes into a dataset
    
    Arguments:
        flows {int} -- Flow based tokens
        n_flows {int} -- Number of flows contained within the dataset
    """
    # UNK = unknown token, variable used for 
    count = [['UNK', -1]]
    count.extend(collections.Counter(flows).most_common(n_flows - 1))
    dictionary = dict()
    for flow, _ in count:
        dictionary[flow] = len(dictionary)
    data = list()
    unk_count = 0
    for flow in flows:
        index = dictrionary.get(flow, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def main():
    make_log_dir('/tmp/test_log/')

if __name__ == "__main__":
    main()