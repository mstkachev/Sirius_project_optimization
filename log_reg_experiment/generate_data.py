import numpy as np
from sklearn.model_selection import train_test_split
import time
import sys
import os
import argparse

from numpy.random import normal, uniform
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix, load_svmlight_file, dump_svmlight_file
from numpy.linalg import norm

from logreg_functions import *
from sigmoid_functions import *
from quadratic_functions import *
from general import *


import sys

import itertools
from scipy.special import binom
from scipy.stats import ortho_group

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file

parser = argparse.ArgumentParser(description='Generate data and provide information about it for workers and parameter server')
parser.add_argument('--n_workers', action='store', dest='n_workers', type=int, default=1, help='Number of workers that will be used')
parser.add_argument('--dataset', action='store', dest='dataset', default='mushrooms', help='The name of the dataset')
parser.add_argument('-b', action='store_true', dest='big_reg', help='Whether to use 1/N regularization or 0.1/N')
parser.add_argument('--dimension', action='store', dest='dimension', type=int, default=300, help='Dimension for generating artifical data')
parser.add_argument('-l', action='store_true', dest='logistic', help='The problem is logistic regression')
parser.add_argument('--scaled', action='store', dest='scaled', type=str, default='non-scaled',
                    help='scaled or non-scaled')
parser.add_argument('--loss_func', action='store', dest='loss_func', type=str, default='quadratic',
                    help='loss function ')

args = parser.parse_args()
n_workers = args.n_workers
dataset = args.dataset
big_reg = args.big_reg
d = args.dimension
scaled = args.scaled
loss_func = args.loss_func

#loss_func = args.loss_func


quadratic_datasets = ["quad_1", "quad_2"]
loss_functions = ["log-reg", "quadratic"]

if loss_func not in loss_functions:
    raise ValueError('wrong loss function')

if loss_func == "quadratic" and dataset not in quadratic_datasets:
    raise ValueError('wrong dataset for quadratic loss function')

if  dataset in quadratic_datasets and loss_func != "quadratic":
    raise ValueError('wrong loss_func for quadratic dataset')

data_name = dataset + ".txt"

user_dir = os.path.expanduser('~/')
RAW_DATA_PATH = os.getcwd() +'/data/'

project_path = os.getcwd() + "/"

data_path = project_path + "data_{0}/".format(dataset)

if not os.path.exists(data_path):
    os.mkdir(data_path)

Xs = []
ys = []

#initial initialization

d = 1000
data_len = d*100
#data matrix
data_dense = np.full((data_len, d), np.nan)

#parameter vector
#it has meaning of "y" labels for logistic regression and parameter "b" for quadratic loss
enc_labels = np.full(data_len, np.nan)

if os.path.isfile(RAW_DATA_PATH + data_name):
    data, labels = load_svmlight_file(RAW_DATA_PATH + data_name)
    enc_labels = labels.copy()
    data_dense = data.todense()
    print (data_dense.shape)
    if loss_func == "log-reg":
        if not np.array_equal(np.unique(labels), np.array([-1, 1], dtype='float')):
            min_label = min(np.unique(enc_labels))
            max_label = max(np.unique(enc_labels))
            enc_labels[enc_labels == min_label] = -1
            enc_labels[enc_labels == max_label] = 1
    if loss_func == "quadratic":
        if dataset == "quad_1":
            assert data_dense.shape == (data_len, d)
        if dataset == "quad_2":
            assert data_dense.shape == (d, d)

else:
    if loss_func=="log-reg":
        raise ValueError("cannot load " + data_name + ".txt")
    elif loss_func=="quadratic":
        if dataset == "quad_1":
            data_dense = np.random.randn(data_len, d)
            enc_labels = np.random.randn(data_len)
            dump_svmlight_file(data_dense, enc_labels, RAW_DATA_PATH + dataset+ ".txt")
        elif dataset == "quad_2":
            data_dense = np.diag(np.arange(1, d+1))
            enc_labels = np.random.randn(d)
            dump_svmlight_file(data_dense, enc_labels, RAW_DATA_PATH + dataset+ ".txt")
        else:
            raise ValueError('wrong dataset for quadratic loss function')
    else:
        raise ValueError('wrong loss function')

#TODO: fix check below
#if any((np.isnan(data_dense))):
 #   raise ValueError("nan values of dataset")

if any(np.isnan(enc_labels)):
    raise ValueError("nan values of labels")

train_feature_matrix, train_labels = data_dense, enc_labels

X = np.array(train_feature_matrix)
y = np.array(train_labels)

if scaled == "scaled":
    scaler = StandardScaler()
    scaler.fit(train_feature_matrix, train_labels)
    X = scaler.transform(train_feature_matrix)

assert len(X.shape)== 2
assert len(y.shape) == 1

data_len = len(train_labels)
train_d = X.shape[0]

#la = np.mean(np.diag(X.T @ X))

d = X.shape[1]
la = 1
w0 = np.random.normal(loc=0.0, scale=1.0, size=d)
M = np.full((d,d), np.nan)
M = init_M (X, loss_func, la)
if loss_func == "log-reg":
    f =    lambda w: logreg_loss(w, X, y, la)
    grad = lambda w: logreg_grad(w, X, y, la)
    result = minimize(fun=f, x0=w0, jac=grad, method="L-BFGS-B", tol=1e-16, options={"maxiter": 10000})
    np.save(data_path + "{0}_clf_coef".format(loss_func), result.x)
    np.save(data_path + "{0}_f_min".format(loss_func), result.fun)
elif loss_func == "quadratic":
    f = lambda w: quadratic_loss(w, X, M, y, la)
    grad = lambda w: quadratic_grad(w, X, M, y, la)
    result = minimize(fun=f, x0=w0, jac=grad, method="L-BFGS-B", tol=1e-16, options={"maxiter": 10000})
    np.save(data_path + "{0}_clf_coef".format(loss_func), result.x)
    np.save(data_path + "{0}_f_min".format(loss_func), result.fun)
else:
    raise ValueError("wrong loss func")

if any(np.isnan(M).flatten()):
    raise ValueError("nan values of M")

print('Number of data points:', data_len)

sep_idx = [0] + [(train_d * i) // n_workers for i in range(1, n_workers)] + [train_d]
# sep_idx = np.arange(0, n_workers * 100 + 1, 100)

data_info = [sep_idx[-1], la]

for i in range(n_workers):
    print('Creating chunk number', i + 1)
    start, end = sep_idx[i], sep_idx[i + 1]
    print(start, end)
    Xs.append(X[start:end])
    ys.append(y[start:end])
    data_info.append(la)


# Save data for master
np.save(data_path + 'X', X)
np.save(data_path + 'y', y)
np.save(data_path + 'data_info', data_info)

# Save data for workers
for worker in range(n_workers):
    np.save(data_path + 'Xs_' + str(worker), Xs[worker])
    np.save(data_path + 'ys_' + str(worker), ys[worker])





