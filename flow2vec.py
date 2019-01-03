from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from configparser import ConfigParser
import generator
import logging
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urillib
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

