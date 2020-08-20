"""
(c) Igor Sokolov

A script for the data preprocessing before launch an algorithm
"""

import numpy as np
from sklearn.model_selection import train_test_split
import time
import sys
import os
import argparse
from numpy.random import normal, uniform
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix, load_svmlight_file, dump_svmlight_file
from numpy.linalg import norm
import itertools
from scipy.special import binom
from scipy.stats import ortho_group
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
from matplotlib import pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from logreg_functions import *

parser = argparse.ArgumentParser(description='Generate data and provide information about it for workers and parameter server')
parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, default=1, help='Number of workers that will be used')
parser.add_argument('--dataset', action='store', dest='dataset', default='mushrooms', help='The name of the dataset')
parser.add_argument('--loss_func', action='store', dest='loss_func', type=str, default='log-reg',
                    help='loss function ')
parser.add_argument('--homogeneous', action='store', dest='is_homogeneous', type=int, default=1, help='Homogeneous or heterogeneous data')

# homogeneous case correspond to the is_homogeneous = 1
# heterogeneous case correspond to the is_homogeneous = 0

args = parser.parse_args()
num_workers = args.num_workers
dataset = args.dataset
loss_func = args.loss_func
is_homogeneous = args.is_homogeneous

assert (loss_func == "log-reg") # our experiments for logistic regression only
assert (is_homogeneous in [0,1])
assert (num_workers >= 1)

# we assume that we will not run for n_workers = 1 in heterogeneous case
if not is_homogeneous:
    if num_workers == 1:
        raise ValueError("num_workers must be more than 1 in heterogeneous case")

data_name = dataset + ".txt"

user_dir = os.path.expanduser('~/')
RAW_DATA_PATH = os.getcwd() +'/data/'

project_path = os.getcwd() + "/"

data_path = project_path + "data_{0}/".format(dataset)

if not os.path.exists(data_path):
    os.mkdir(data_path)

#TODO: assert these values below
enc_labels = np.nan
data_dense = np.nan

if os.path.isfile(RAW_DATA_PATH + data_name):
    data, labels = load_svmlight_file(RAW_DATA_PATH + data_name)
    enc_labels = labels.copy()
    data_dense = data.todense()
    if not np.array_equal(np.unique(labels), np.array([-1, 1], dtype='float')):
        min_label = min(np.unique(enc_labels))
        max_label = max(np.unique(enc_labels))
        enc_labels[enc_labels == min_label] = -1
        enc_labels[enc_labels == max_label] = 1
    print (enc_labels.shape, enc_labels[-5:])
else:
    raise ValueError("cannot load " + data_name + ".txt")

assert (type(data_dense) == np.matrix or type(data_dense) == np.ndarray)
assert (type(enc_labels) == np.ndarray)

if np.sum(np.isnan(enc_labels)) > 0:
    raise ValueError("nan values of labels")

if np.sum(np.isnan(data_dense)) > 0:
    raise ValueError("nan values in data matrix")

#assert (any(np.isnan(enc_labels)))
#assert (any(np.isnan(data_dense)))

print ("Data shape: ", data_dense.shape)

#for homogeneus case X is a 2d array
#for heterogeneus case X is a 3d array

X = np.nan
y = np.nan

if is_homogeneous:
    X = data_dense
    y = enc_labels
    assert len(X.shape) == 2
    assert len(y.shape) == 1
    data_len = len(enc_labels)
    train_d = X.shape[0]
else:
    #Maxim's code here
    raise NotImplementedError

#assert ((X,y) == (None, None))

if np.sum(np.isnan(y)) > 0:
    raise ValueError("nan values of labels")

if np.sum(np.isnan(X)) > 0:
    raise ValueError("nan values in data matrix")

la = 0.1 #regularization parameter

data_info = [la]

# Save data
np.save(data_path + 'X', X)
np.save(data_path + 'y', y)
np.save(data_path + 'data_info', data_info)





