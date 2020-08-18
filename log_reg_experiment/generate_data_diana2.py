import numpy as np
import time
import sys
import os
import argparse

from numpy.random import normal, uniform
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix, load_svmlight_file
from numpy.linalg import norm

parser = argparse.ArgumentParser(description='Generate data and provide information about it for workers and parameter server')
parser.add_argument('--n_workers', action='store', dest='n_workers', type=int, default=4, help='Number of workers that will be used')
parser.add_argument('--dataset', action='store', dest='dataset', default='real-sim', help='The name of the dataset')
parser.add_argument('-b', action='store_true', dest='big_reg', help='Whether to use 1/N regularization or 0.1/N')
parser.add_argument('--dimension', action='store', dest='dimension', type=int, default=300, help='Dimension for generating artifical data')
parser.add_argument('-l', action='store_true', dest='logistic', help='The problem is logistic regression')

args = parser.parse_args()
n_workers = args.n_workers
dataset = args.dataset
big_reg = args.big_reg
d = args.dimension
logistic = True

user_dir = os.path.expanduser('~/')
SCRIPTS_PATH = '/Users/mishchk/Downloads/diana2/data/'
DATA_PATH = '/Users/mishchk/experiments/datasets/'

zero_based = {'mushrooms': False, 'a5a': False, 'a8a': False}

def generate_data(d, min_cond=1e2, max_cond=1e4, diagonal=False):
    if diagonal:
        X = np.diag(uniform(low=10, high=1e3, size=d))
    else:
        ratio = np.inf
        while (ratio < min_cond) or (ratio > max_cond):
            X = 10 * make_spd_matrix(n_dim=d)
            vals, _ = np.linalg.eig(X)
            ratio = max(vals) / min(vals)
        print(ratio)
    y = 10 * uniform(low=-1, high=1, size=d)
    return X, y
    
def load_data(data_name):
    data = load_svmlight_file(DATA_PATH + data_name, zero_based=zero_based.get(dataset, 'auto'))
    return data[0], data[1]

Xs = []
ys = []

X, y = load_data(dataset)
X = X[:3000]
y = y[:3000]
data_len = len(y)
print('Number of data points:', data_len)
sep_idx = [0] + [(data_len * i) // n_workers for i in range(1, n_workers)] + [data_len]
# sep_idx = np.arange(0, n_workers * 100 + 1, 100)
L = 0.25# * np.max(X.multiply(X).sum(axis=1)) if logistic else np.max(np.absolute(np.linalg.eigvals((X.T @ X).toarray())))
data_info = [sep_idx[-1], L]
for i in range(n_workers):
    print('Creating chunk number', i + 1)
    start, end = sep_idx[i], sep_idx[i + 1]
    print(start, end)
    if logistic:
        if 2 in y:
            y = 2 * y - 3
        Xs.append(X[start:end].todense())
        ys.append(y[start:end])
        data_info.append(0.25)
            
# Remove old data
# os.system("bash -c 'rm {0}/Xs*'".format(SCRIPTS_PATH))
# os.system("bash -c 'rm {0}/ys*'".format(SCRIPTS_PATH))

# Save data for master
np.save(SCRIPTS_PATH + 'X', X.todense())
np.save(SCRIPTS_PATH + 'y', y)

# Save data for workers
for worker in range(n_workers):
    np.save(SCRIPTS_PATH + 'Xs_' + str(worker), Xs[worker])
    np.save(SCRIPTS_PATH + 'ys_' + str(worker), ys[worker])
    np.save(SCRIPTS_PATH + 'data_info', data_info)