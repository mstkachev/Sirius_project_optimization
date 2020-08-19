"""
(c) Igor Sokolov
experiment for logistic regression
based on the paper https://arxiv.org/abs/1912.06036v1

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

from numpy.random import normal, uniform
from numpy.linalg import norm
from general import *

parser = argparse.ArgumentParser(description='Run NSYNC algorithm')
parser.add_argument('--max_it', action='store', dest='max_it', type=int, default=None, help='Maximum number of iterations')

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

max_it = args.max_it

batch_size = args.batch_size
num_workers = args.num_workers
epoch_size = args.epoch_size
num_local_steps = args.num_local_steps
dataset = args.dataset
is_continue = args.is_continue #means that we want (or do not want) to continue previously started experiments

launch_number = args.launch_number
tolerance = args.tolerance

NUM_GLOBAL_STEPS = 1000 #every NUM_GLOBAL_STEPS times of communication we will store our data

#first_data_save = 1 # required flag when we save data first time

loss_func = "log-reg"

# #debug only #########
#
#
# dataset = "mushrooms"
# batch_size = 50
# max_it =812400000
#
#
# ###########

convergense_eps = tolerance


if max_it is None:
    max_it = np.inf

assert (batch_size >= 1)
assert (num_workers >= 1)
assert (num_local_steps >= 1)
assert (epoch_size >= 1)
assert (launch_number >= 1)
assert (convergense_eps >0 )
assert (is_continue in [0,1])


def myrepr(x):
    return repr(round(x, 2)).replace('.',',') if isinstance(x, float) else repr(x)


######################


def init_stepsize(X, la, num_local_steps):
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
    return 1/(8 * num_local_steps * L)

def init_epoch_size(X, batch_size):
    n, d = X.shape
    temp = int(n / batch_size)
    return temp if temp % num_local_steps == 0 else temp - (temp % num_local_steps) + num_local_steps

def continue_criterion (it, max_it, convergense_eps, f_grad_norm):
    return it < max_it and f_grad_norm < convergense_eps

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
    w_0, f_grad_0, loss, f_grad_norms, its_comm, epochs,  W_prev, V_prev = None,None, None,None,None, None, None, None
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

    if f_grad_norms is None:
        f_grad_norms = [np.linalg.norm(x=f_grad_0,ord=2)]

    W_prev = np.repeat(w_0[np.newaxis, :], num_workers, axis=0)
    V_prev = np.repeat(f_grad_0[np.newaxis, :], num_workers, axis=0)

    #TODO: fix assert below
    assert ( (w_0, f_grad_0, loss, f_grad_norms, its_comm, epochs,  W_prev, V_prev) == (None,None, None,None,None, None, None, None))
    assert (W_prev.shape == (num_workers, d) and V_prev.shape == (num_workers, d))

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

experiment_name = "local_spider_homo_"

experiment = '{0}_{1}_{2}_{3}_{4}'.format(experiment_name, batch_size, num_workers, num_local_steps, epoch_size)

logs_path = project_path + "logs{2}_{0}_{1}/".format(dataset, experiment, launch_number)
data_path = project_path + "data_{0}/".format(dataset)

if not os.path.exists(logs_path):
    os.makedirs(logs_path)

if not os.path.exists(data_path):
    os.mkdir(data_path)

data_info = np.load(data_path + 'data_info.npy')
N, la = data_info[:2]

X = np.load(data_path + 'X.npy')
y = np.load(data_path + 'y.npy')
N_X, d = X.shape

data_length_total = N_X

currentDT = datetime.datetime.now()
print (currentDT.strftime("%Y-%m-%d %H:%M:%S"))
print (experiment)

step_size = init_stepsize(X, la )
if epoch_size is None:
    epoch_size = init_epoch_size(X, batch_size)

it_comm = 0 # current iteration of communication

w_avg, loss, f_grad_norms, its_comm, epochs,  W_prev, V_prev = init_estimates (X, y, la, num_workers, is_continue, experiment, logs_path, loss_func)

global_it = 0 #iterator of while loop

while continue_criterion (it, max_it, convergense_eps, f_grad_norms[-1]):

    #TODO: think about initialization before epochs

    W = W_prev - step_size * V_prev # do a step for all workers (5th)

    for t in range(epoch_size):

        i_batch =  np.random.choice(data_length_total, batch_size) #generate uniformly subset
        V = sample_matrix_logreg_sgrad(W, X, y, la, i_batch) - sample_matrix_logreg_sgrad(W_prev, X, y, la, i_batch)  + V_prev # (7th)

        if t % num_local_steps ==0:
            w_avg = np.mean(W, axis=0)  #(9th)
            v_avg = np.mean(V, axis=0)  #(10th)
            W_prev = np.repeat(w_avg[np.newaxis, :], num_workers, axis=0)#clone averaged point to each worker (broadcast)
            V_prev = np.repeat(v_avg[np.newaxis, :], num_workers, axis=0)#clone averaged gradient estimation to each worker (broadcast)

            #(below)save current state of the iteration process
            it_comm += 1

            f_grad_norms.append(np.linalg.norm(x=logreg_grad(w_avg, X, y,la),ord=2))
            its_comm.append(it_comm)
            #ws_avg.append(w_avg)
            loss.append(logreg_loss(w_avg, X, y, la))
            epochs.append(global_it + t/epoch_size)

            if it_comm % NUM_GLOBAL_STEPS == 0:
                print ("it_comm: {0} , f_grad_norm: {1}".format(it_comm, f_grad_norms[-1]))
                #TODO: implement function below
                save_data(loss, f_grad_norms, its_comm, epochs, w_avg, logs_path, experiment)

        W = W_prev - step_size * V_prev  # do a step for all workers (12th)

    w_avg = np.mean(W, axis=0)  # (15th)
    W_prev = np.repeat(w_avg[np.newaxis, :], num_workers, axis=0)  # (16th)
    f_grad = logreg_grad(w_avg, X, y,la)
    V_prev =  np.repeat(f_grad[np.newaxis, :], num_workers, axis=0)#(17th)
    global_it += 1


##

save_data(loss, f_grad_norms, its_comm, epochs, w_avg, logs_path, experiment)

#print("There were done", len(its), "iterations")





