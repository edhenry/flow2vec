from __future__ import absolute_import, division, print_function

import collections
import logging
import os
import pickle
import random
import typing
from pathlib import Path

import numpy as np
import pandas as pd

def save_dataset(dir: str, filename: str, dataset: pd.Series, logger: logging.Logger):
    """Location on filesystem to save parameterized subset of training data
    
    Currently supporting plain old python pickling with the option to support
    more efficient serialization formats in the future

    Arguments:
        dir {str} -- directory on local filesystem to save dataset
        filename {str} -- name of the filename of the dataset
    """
    fn = (dir + filename) # type: str
    
    dataset_dir = Path(dir)

    if dataset_dir.exists():
        logger.info(f"Dataset directory {dataset_dir} already exists!")
        return
    else:
        os.makedirs(dataset_dir)
        logger.info(f"Dataset directory {dataset_dir} has been created!")

        os.chdir(dataset_dir)
        with open(filename, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

def stringify(dataframes: typing.List[pd.DataFrame]) -> typing.List[pd.Series]:
    """Convert rows of binetflow file to strings for use in tokenizing
    
    Arguments:
        dataframe {pd.DataFrame} -- pandas dataframe containing netflow records
    
    Returns:
        List[pd.Series] -- list of pandas Series object containing stringified netflow records
    """

    string_vals_list = [] # type: list

    for df in dataframes:
        string_vals = df.stack().groupby(level=0).apply(','.join)
        string_vals_list.append(string_vals)
    
    return string_vals_list

def count_tokens(series_list: typing.List[pd.Series]) -> collections.Counter:
    """Count number of unique tokens in our dataset
    
    Arguments:
        series_list {typing.List[pd.Series]} -- list of unique tokens in our dataset
    
    Returns:
        collections.Counter -- Counter object of all unique values and their counts within a list of series.
    """

    counts = collections.Counter() # type: dict
    for series in series_list:
        for flow in series:
            counts[flow] += 1
    
    return counts

def create_corpora(dataframe: pd.DataFrame, window: int, corpus_count: int):
    """Create corpora of network flows for use in training a model
    
    Arguments:
        dataframe {pd.DataFrame} -- DataFrame to split into corpora
        window {int} -- window size
        corpus_count {int} -- how many corpora to create
    
    Returns:
        [list] -- array of corpora (corpus)
    """    
    corpus = [] # type: List[dataframe]
    corpora = []
    beginning = 0
    end = window
    for i in range(corpus_count):
        corpus = dataframe.iloc[beginning:end]
        corpora.append(corpus)
        beginning = end + 1
        end += window
    return corpora