from __future__ import absolute_import, division, print_function

import argparse
import collections
import logging
import math
import os
import pathlib
import pprint
import random
import sys
import typing
import zipfile
from configparser import ConfigParser
from tempfile import gettempdir

import numpy as np
import pandas as pd
import tensorflow as tf
from six.moves import urllib, xrange
from tensorflow.contrib.tensorboard.plugins import projector

import generator
import log
import utility

config_file = "./sample_config.ini"
cp = ConfigParser()
cp.read(config_file)

log_dir = cp["DEFAULT"].get("log_dir")
experiment_dir = cp["DEFAULT"].get("experiment_dir")
exp_dataset_dir = cp["DEFAULT"].get("exp_dataset_dir")
flow_files = cp["DEFAULT"].get("flow_files").split(",") # type: List[str]
num_flows = cp["DEFAULT"].get("num_flows") # type: int
window_size = cp["DEFAULT"].get("window_size") # type: int
num_windows = cp["DEFAULT"].get("num_windows") # type: int

# training variables -- see defintions of variables within the config.ini file
batch_size = cp["TRAIN"].get("batch_size") # type: int
embedding_size = cp["TRAIN"].get("embedding_size") # type: int
skip_window = cp["TRAIN"].get("skip_window") # type: int
over_sample_rate = cp["TRAIN"].get("over_sample_rate") # type: int
num_negative_examples = cp["TRAIN"].get("num_negative_examples") # type: int
validation_size = cp["TRAIN"].get("validation_size") # type: int
validation_window = cp["TRAIN"].get("validation_window") # type: int
validation_examples = np.random.choice(validation_window, validation_size, replace=False)


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

def main():

    logs = log.logger() # instantiate logger
    log.make_log_dir(log_dir, logs) # create logging dir

    flows = generator.dataframe(flow_files)
    categories, labels, _ = generator.split_cols(flows)
    corpora = utility.create_corpora(categories, 25, 1000)
    # TODO implement caching feature for dataset versioning for the many types of subsampling and windowing that
    # may be performed over the dataset. This will help with reprodicibility in the future

    # TODO research the unique values contained within each column of the flows
    # would be useful to produce histograms for this for analysis    
    strings = utility.stringify(corpora)
    token_counts = utility.count_tokens(strings)

    vocabulary_size = len(token_counts.keys())
    
    # Save dataset for future use
    # Eventually would like to utilize something like Pachyderm to allow
    # for data versioning per experimental runpip
    utility.save_dataset((experiment_dir + exp_dataset_dir), "test.pkl", strings, logs)

        

if __name__ == "__main__":
    main()
