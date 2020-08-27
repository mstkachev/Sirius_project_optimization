"""
(c) Igor Sokolov
https://github.com/mstkachev/Sirius_project_optimization

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

#debug section

"""
num_workers = 20
dataset = 'a9a'
loss_func = 'log-reg' 
is_homogeneous = 0

assert (loss_func == "log-reg") # our experiments for logistic regression only
assert (is_homogeneous in [0,1])
assert (num_workers >= 1)
"""

def nan_check (lst):
    """
    Check whether has any item of list np.nan elements
    :param lst: list of datafiles (eg. numpy.ndarray)
    :return:
    """
    for i, item in enumerate (lst):
        if np.sum(np.isnan(item)) > 0:
            raise ValueError("nan files in item {0}".format(i))

# we assume that we will not run for num_workers = 1 in heterogeneous case
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
#for heterogeneus case X is a list of 2d arrays

X = np.nan
y = np.nan

la = 0.1 #regularization parameter
data_info = [la]


if is_homogeneous:
    X = data_dense
    y = enc_labels
    assert len(X.shape) == 2
    assert len(y.shape) == 1
    data_len = len(enc_labels)
    train_d = X.shape[0]
    nan_check([X,y])
    np.save(data_path + 'X', X)
    np.save(data_path + 'y', y)
else:
    data_dense = np.hstack((data_dense, enc_labels[:, np.newaxis])) # hstack y-label to the dataset
    A = data_dense[np.where(enc_labels == - 1)]
    B = data_dense[np.where(enc_labels == 1)]
    K = num_workers  # количество рабочих, заменить имя справа, если я не угадал
    C = []
    z = np.zeros((0, data_dense.shape[1]))
    for i in range(K):
        C.append(z)
    h = 2 * A.shape[0] / ((K - 1) * K)
    h = max(int(h), 1)
    # print(h)
    hh = 2 * B.shape[0] / ((K - 1) * K)
    hh = max(int(hh), 1)
    # print(hh)
    for i in range(1, K - 1):
        C[i] = np.vstack((C[i], A[int((h * i * (i - 1) / 2)): int((h * i * (i + 1) / 2))]))
        C[-i - 1] = np.vstack((C[-i - 1], B[int((hh * i * (i - 1) / 2)): int((hh * i * (i + 1) / 2))]))
    C[K - 1] = np.vstack((C[K - 1], A[int((h * (K - 1) * (K - 2) / 2)):]))
    C[0] = np.vstack((C[0], B[int((hh * (K - 1) * (K - 2) / 2)):]))

    list(map(np.random.shuffle, C))

    y = [np.squeeze(np.asarray(C[i][:, -1])) for i in range(len(C))]
    X = [C[i][:, :-1] for i in range(len(C))]  # remove y-label from the dataset

    nan_check(y)
    nan_check(X)
    for i in range (len(X)):
        print ("worker {0} has {1} datasamples and {2} labels".format(i, X[i].shape[0], y[i].shape[0]))
        np.save(data_path + 'X_{0}_nw{1}_{2}'.format(dataset, num_workers, i), X[i])
        np.save(data_path + 'y_{0}_nw{1}_{2}'.format(dataset, num_workers, i), y[i].flatten())
        np.save(data_path + 'data_info', data_info)

np.save(data_path + 'data_info', data_info)
# Save data






