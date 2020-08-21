"""
(c) Igor Sokolov
experiment for logistic regression
based on the paper

(i th) as a comments means that author refers to the i-th line of the original algorithm of the paper

W_prev - workers matrix of current point
V_prev - workers matrix of current vector v (each row correspond to the particular worker)

"""

import numpy as np
import time
import argparse
import datetime
import scipy
import subprocess, os, sys

from contextlib import redirect_stdout

from scipy.linalg import eigh

from numpy.random import normal, uniform
from numpy.linalg import norm

from logreg_functions import *


parser = argparse.ArgumentParser(description='Run NSYNC algorithm')
parser.add_argument('--max_epochs', action='store', dest='max_epochs', type=int, default=None, help='Maximum number of epochs')
parser.add_argument('--max_num_comm', action='store', dest='max_num_comm', type=int, default=None, help='Maximum number of communications')
parser.add_argument('--max_it', action='store', dest='max_it', type=int, default=None, help='Maximum number of iteration')

parser.add_argument('--batch_size', action='store', dest='batch_size', type=int, default=1, help='Minibatch size')
parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, default=10, help='Number of workers')
parser.add_argument('--epoch_size', action='store', dest='epoch_size', type=int, default=None, help='Epoch size')
parser.add_argument('--num_local_steps', action='store', dest='num_local_steps', type=int, default=10, help='Number of local steps for each client')
parser.add_argument('--continue', action='store', dest='is_continue', type=int, default=0, help='Continue or restart')
parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='mushrooms',
                    help='Dataset name for saving logs')
parser.add_argument('--launch_number', action='store', dest='launch_number', type=int, default=1, help='launch_number')
parser.add_argument('--tol', action='store', dest='tolerance', type=float, default=1e-12, help='tolerance')


args = parser.parse_args()

max_num_comm = args.max_num_comm
max_epochs = args.max_epochs
max_it = args.max_it

batch_size = args.batch_size
num_workers = args.num_workers
epoch_size = args.epoch_size
num_local_steps = args.num_local_steps
dataset = args.dataset
is_continue = args.is_continue #means that we want (or do not want) to continue previously started experiments

launch_number = args.launch_number
tolerance = args.tolerance


#debug section
"""

max_epochs = None
max_num_comm = None
#max_epochs = 100
max_it = 1000
batch_size = 20
num_workers = 20
epoch_size = None
num_local_steps = 10
dataset = "a9a"
is_continue = 0 #means that we want (or do not want) to continue previously started experiments
launch_number = 1
tolerance = 1e-12
"""

NUM_GLOBAL_STEPS = 1000 #every NUM_GLOBAL_STEPS times of communication we will store our data

#first_data_save = 1 # required flag when we save data first time

loss_func = "log-reg"

convergense_eps = tolerance


if max_epochs is None:
    max_epochs = np.inf

if max_num_comm is None:
    max_num_comm = np.inf

if max_it is None:
    max_it = np.inf

assert (batch_size >= 1)
assert (num_workers >= 1)
assert (num_local_steps >= 1)
if epoch_size is not None:
    assert (epoch_size >= 1)
assert (launch_number >= 1)
assert (convergense_eps >0 )
assert (is_continue in [0,1])


def myrepr(x):
    return repr(round(x, 2)).replace('.',',') if isinstance(x, float) else repr(x)

######################

def init_stepsize(X, la, num_local_steps, batch_size):
    """
    Returns stepsize
    :param X: data matrix
    :param la: regularization parameter
    :param num_local_steps: num_local_steps
    :return: stepsize
    """
    n, d = X.shape
    la_max = scipy.linalg.eigh(a=(X.T @ X), eigvals_only=True, turbo=True, type=1, eigvals=(d - 1, d - 1))
    L = (1 / (4 * n)) * la_max + la * 2 #lipshitz constant

    #return 1/(8 * num_local_steps * L)
    #return np.sqrt(batch_size) / (np.sqrt(n) * L)
    #print (1 /(2*L))
    return 1 /(25*L)
    #return 0.0005  # homo case

def init_epoch_size(X, batch_size):
    n, d = X.shape
    return int(n / batch_size)

def continue_criterion (it, max_epochs, convergense_eps, f_grad_norm):
    return it < max_epochs and f_grad_norm < convergense_eps

def load_data (experiment, logs_path):
    """
    Returns data files assuming that they exist
    :param experiment: string indicating particular experiment
    :param logs_path: path to the folder with logs
    :return: data files
    """
    if os.path.isfile(logs_path + 'solution' + '_' + experiment + ".npy"):
        w_0 = np.array(np.load(logs_path + 'solution' + '_' + experiment + ".npy"))
    else:
        raise ValueError("cannot upload start-point")

    if os.path.isfile(logs_path + 'loss' + "_" + experiment + ".npy"):
        loss = np.load(logs_path + 'loss' + '_' + experiment + ".npy")
    else:
        raise ValueError("cannot load loss info")

    if os.path.isfile(logs_path + 'norms' + "_" + experiment + ".npy"):
        f_grad_norms = np.load(logs_path + 'norms' + '_' + experiment + ".npy")
    else:
        raise ValueError("cannot load norms info")

    if os.path.isfile(logs_path + 'communication' + "_" + experiment + ".npy"):
        its_comm = np.load(logs_path + 'communication' + '_' + experiment + ".npy")
    else:
        raise ValueError("cannot load communication info")

    if os.path.isfile(logs_path + 'epochs' + "_" + experiment + ".npy"):
        epochs = np.load(logs_path + 'epochs' + '_' + experiment + ".npy")
    else:
        raise ValueError("cannot load epochs info")

    return w_0, list(loss), list(f_grad_norms), list(its_comm), list(epochs)

def nan_check (lst):
    """
    Check whether has any item of list np.nan elements
    :param lst: list of datafiles (eg. numpy.ndarray)
    :return:
    """
    for i, item in enumerate (lst):
        if np.sum(np.isnan(item)) > 0:
            raise ValueError("nan files in item {0}".format(i))

def init_estimates(X, y, la, num_workers, is_continue, experiment, logs_path, loss_func):
    """
    Returns initial esrimates of variable parameters
    :param X:
    :param y:
    :param la:
    :param num_workers:
    :param is_continue:
    :param experiment:
    :param logs_path:
    :param loss_func:
    :return:
    """
    w_0, f_grad_0, loss, f_grad_norms, its_comm, epochs,  W_prev, V_prev= np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    N_X, d = X.shape

    if is_continue:
        #then load existing data
        w_0, loss, f_grad_norms, its_comm, epochs = load_data(experiment, logs_path)
    else:
        if not os.path.isfile(data_path + 'w_init_{0}.npy'.format(loss_func)):
            # create a new w_0
            w_0 = np.random.normal(loc=0.0, scale=1.0, size=d)
            np.save(data_path + 'w_init_{0}.npy'.format(loss_func), w_0)
            w_0 = np.array(np.load(data_path + 'w_init_{0}.npy'.format(loss_func)))
        else:
            # load existing w_0
            w_0 = np.array(np.load(data_path + 'w_init_{0}.npy'.format(loss_func)))

        loss = [logreg_loss(w_0, X, y, la)]
        its_comm = [0]
        epochs = [0]

    f_grad_0 = logreg_grad(w_0, X, y, la)

    if np.sum(np.isnan(f_grad_norms)) > 0:
        f_grad_norms = [np.linalg.norm(x=f_grad_0, ord=2)]

    W_prev = np.repeat(w_0[np.newaxis, :], num_workers, axis=0)

    V_prev = np.zeros(d)

    #check whether data initialized
    nan_check([w_0, f_grad_0, loss, f_grad_norms, its_comm, epochs,  W_prev, V_prev])

    assert (W_prev.shape == (num_workers, d))

    return w_0, loss, f_grad_norms, its_comm, epochs,  W_prev, V_prev

def save_data(loss, f_grad_norms, its_comm, epochs, w_avg, logs_path, experiment):
    np.save(logs_path + 'loss' + '_' + experiment, np.array(loss))
    np.save(logs_path + 'communication' + '_' + experiment, np.array(its_comm))
    np.save(logs_path + 'epochs' + '_' + experiment, np.array(epochs))
    np.save(logs_path + 'solution' + '_' + experiment, w_avg)
    np.save(logs_path + 'norms' + '_' + experiment, np.array(f_grad_norms))

#######################


user_dir = os.path.expanduser('~/')

project_path = os.getcwd() + "/"

experiment_name = "local_fed-adam_homo"

experiment = '{0}_{1}_{2}_{3}'.format(experiment_name, batch_size, num_workers, num_local_steps)

logs_path = project_path + "logs{2}_{0}_{1}/".format(dataset, experiment, launch_number)
data_path = project_path + "data_{0}/".format(dataset)

if not os.path.exists(logs_path):
    os.makedirs(logs_path)

if not os.path.exists(data_path):
    os.mkdir(data_path)

data_info = np.load(data_path + 'data_info.npy')
la = data_info[0]

#print ("la: ", la, type(la))

assert (type(la) == np.float64)

X = np.load(data_path + 'X.npy')
y = np.load(data_path + 'y.npy')
N_X, d = X.shape

data_length_total = N_X

currentDT = datetime.datetime.now()
print (currentDT.strftime("%Y-%m-%d %H:%M:%S"))
print (experiment)

#TODO: init stepsize
step_size = init_stepsize(X, la, num_local_steps, batch_size)

#TODO: init epochs
if epoch_size is None:
    epoch_size = init_epoch_size(X, batch_size)

w_avg, loss, f_grad_norms, its_comm, epochs,  W_prev, V_prev = init_estimates (X, y, la, num_workers, is_continue, experiment, logs_path, loss_func)

beta1 = 0.9
beta2 = 0.99
delta = 1e-8

it_comm = its_comm[-1] # current iteration of communication
it = 0
epoch_it = 0 #iterator of while loop

W_K = W_prev #W_K is the W after num_local_steps iter
D_prev = np.zeros (W_prev.shape)

while it < max_it and epoch_it < max_epochs and its_comm[-1] < max_num_comm and f_grad_norms[-1] > convergense_eps:
    it += 1

    #TODO:think about stepsize
    #step_size = np.sqrt(data_length_total/it)

    batch_list = [np.random.choice(data_length_total, batch_size) for i in range(num_workers)] #generate uniformly subset
    f_sgrad_matrix = sample_matrix_logreg_sgrad(W_prev, X, y, la, batch_list)

    W = W_prev - step_size * f_sgrad_matrix # do a step for all workers

    if it % num_local_steps == 0:
        D = W - W_K

        d_avg = np.mean(D, axis=0)
        D = np.repeat(d_avg[np.newaxis, :], num_workers, axis=0)  # clone averaged point to each worker (broadcast)

        D = beta1*D_prev + (1 - beta1)*D

        V = beta2 * V_prev + (1 - beta2) * D ** 2

        W_K = W_K + (step_size/(np.sqrt(V) + delta))*D
        V_prev = V

        #(below)save current state of the iteration process
        w_avg = np.mean(W_K, axis=0)
        it_comm += 1
        f_grad_norms.append(np.linalg.norm(x=logreg_grad(w_avg, X, y,la),ord=2))
        its_comm.append(it_comm)
        #ws_avg.append(w_avg)
        loss.append(logreg_loss(w_avg, X, y, la))
        epochs.append(it * batch_size/epoch_size)

        if it_comm % int(NUM_GLOBAL_STEPS/100) == 0:
            print("{4}, it: {5}, epoch_it: {3}, it_comm: {0} , epoch: {1}, f_grad_norm: {2}".format(it_comm, round (epochs[-1],4), round (f_grad_norms[-1],4),epoch_it, experiment, it))
        if it_comm % NUM_GLOBAL_STEPS == 0:
            #TODO: implement function below
            save_data(loss, f_grad_norms, its_comm, epochs, w_avg, logs_path, experiment)
    W_prev = W

##

save_data(loss, f_grad_norms, its_comm, epochs, w_avg, logs_path, experiment)

#print("There were done", len(its), "iterations")





