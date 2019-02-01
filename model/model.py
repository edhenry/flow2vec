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


class Model():

    def __init__(self, batch_size: int, embedding_size: int, 
                 skip_window: int, over_sample_rate: int, 
                 num_negative_examples: int, validation_size: int, 
                 validation_window: int, validation_examples: np.ndarray):

                self.batch_size = batch_size
                self.embedding_size = embedding_size
                self.skip_window = skip_window
                self.over_sample_rate = over_sample_rate
                self. num_negative_examples = num_negative_examples
                self.validation_size = validation_size
                self.validation_window = validation_window
                
        
                graph = tf.Graph()

                with graph.as_default():
                    with tf.name_scope("inputs"):
                        training_inputs = tf.placeholder(tf.int32, [batch_size])
                        training_labels = tf.placeholder(tf.int32, [batch_size, 1])
                        validation_dataset = tf.constant(valid_examples, dtype=tf.int32)
