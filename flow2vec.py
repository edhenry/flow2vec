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

flow_files = cp["DEFAULT"].get("flow_files").split(",")
num_flows = cp["DEFAULT"].get("num_flows")

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
        index = dictionary.get(flow, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def top_n(n: int, tokens: list) -> list:
    """Return a list of top-n unique elements of the dataset
    
    Arguments:
        n {int} -- number of top unique elements to return
    
    Returns:
        list -- array of top-n elements within a dataset
    """
    top_n_elements = tokens[:n]
    return top_n_elements



def main():
    make_log_dir('/tmp/test_log/')

    flows = generator.dataframe(flow_files)
    categories, labels, _ = generator.split_cols(flows)
    corpora = generator.create_corpora(categories, 25, 10)
    # TODO implement caching feature for dataset versioning for the many types of subsampling and windowing that
    # may be performed over the dataset. This will help with reprodicibility in the future

    # TODO research the unique values contained within each column of the flows
    # would be useful to produce histograms for this for analysis
    print(corpora[0].set_index(['Proto', 'SrcAddr']).count(level='Proto'))

if __name__ == "__main__":
    main()