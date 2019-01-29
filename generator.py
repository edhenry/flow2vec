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

# All preprocessing steps that are defined within this module are with
# the understanding that these flow files are encoded in the .binetflow
# file format that is available within the Argus netflow utility.
# https://www.qosient.com/argus/argusnetflow.shtml

# The dataset that was used in this research effort is the CTU-13 dataset.
# Included in the provided link is an explanation of the dataset along with 
# an explanation of the features that are available within the dataset.

def strip(text):
    """Strip white space from text
    
    Arguments:
        text {string} -- string of text to strip white space from
    
    Returns:
        string
    """

    return text.strip()

# Work left to be done on the potential preprocessing
# steps that can be used for feature engineering
def sort_ip_flow(df: pd.DataFrame, ip: str) -> dict:
    """Match IP against a flow srcIP
    
    Arguments:
        ip {string} -- string representation of an IP address (e.g. 192.168.1.1)
    """
    flow_list = []
    for flow in df:
        if ip == flow[1][3]:
            flow_list.append(flow)
    return {ip: flow_list}

# Legacy hashing functions, might not be useful anymore
# though it might be useful later on so I will keep them
# in the code for now
def process_flow(flow):
    """Create tokens of flow data
    
    Arguments:
        flow {[type]} -- [description]
    """
    # create hashes of values
    proto_hash = hasher(flow[1][2])        
    srcip_hash = hasher(flow[1][3])        
    srcprt_hash = hasher(flow[1][4]) 
    dstip_hash = hasher(flow[1][6])    
    dstprt_hash = hasher(flow[1][7]) 
    flow_list = list(flow)       
    # Insert hashes as entry in tuple for each flow
    flow_list.insert(4, (str(proto_hash), str(srcip_hash), str(srcprt_hash), 
                         str(dstip_hash), str(dstprt_hash)))    
    # Re-cast flow entry as tuple w/ added hash tuple
    flow = tuple(flow_list)
    return(flow)

def dataframe(filenames: list):
    """[summary]
    
    Arguments:
        filename {str} -- [description]
    """
    flowdata = pd.DataFrame()

    for file in filenames:
        frame = pd.read_csv(file, sep=',', header=0)
        flowdata = flowdata.append(frame, ignore_index=True)
    
    flowdata.rename(columns=lambda x: x.strip(), inplace=True)
    return flowdata

    
def split_cols(dataframe: pd.DataFrame):
    """Subsample a dataframe of netflow data and return a tuple of
    subsampled data, labels, and a combination dataframe of both as well
    
    Arguments:
        dataframe {pd.DataFrame} -- [description]
    
    Returns:
        [type] -- [description]
    """

    categories = dataframe.loc[:,['Proto', 'SrcAddr', 'DstAddr','Dport']]
    labels = dataframe.loc[:,['Label']]

    categories_and_labels = dataframe.loc[:,['Proto', 'SrcAddr', 'DstAddr',
                                              'Dport', 'Label']]
    
    return categories, labels, categories_and_labels


def generate_batch(batch_size: int, num_skips: int, skip_window: int):
    """Generate a batch for training
    
    Arguments:
        batch_size {int} -- batch_size for training dataset
        num_skips {int} -- number of skips 
        skip_window {int} -- size of window of surrounding tokens
    """
    global data_index
    data_index = 0
    # match these parameters to initial word2vec window
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_words in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_words]
        if data_index == len(data):
            buffer.extend(data[0:span])
    return