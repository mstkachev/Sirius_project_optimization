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
parser.add_argument('--max_it', action='store', dest='max_it', type=int, help='Maximum number of iterations')
parser.add_argument('--max_t', action='store', dest='max_t', type=float, help='Time limit')

parser.add_argument('--batch', action='store', dest='batch', type=int, default=1, help='Minibatch size')

parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, default=10, help='Number of workers')

parser.add_argument('--num_local_steps', action='store', dest='num_local_steps', type=int, default=10, help='Number of local steps for each client')

parser.add_argument('--continue', action='store', dest='is_continue', type=int, default=0, help='Continue or restart')

parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='mushrooms',
                    help='Dataset name for saving logs')

parser.add_argument('--launch_number', action='store', dest='launch_number', type=int, default=1, help='launch_number')

parser.add_argument('--tol', action='store', dest='tolerance', type=float, default=1e-12, help='tolerance')

args = parser.parse_args()

max_it = args.max_it
max_t = args.max_t

batch = args.batch
num_workers = args.num_workers
num_local_steps = args.num_local_steps

dataset = args.dataset

step_size = args.step_size
launch_number = args.launch_number

tolerance = args.tolerance

# #debug only #########
#
#
# dataset = "mushrooms"
# sampling_kind = "uniform"
# batch = 50
# max_it =812400000
# scaled ="non-scaled"
# step_type = "optimal"
#
#
# ###########

convergense_eps = tolerance


if max_it is None:
    max_it = np.inf
if max_t is None:
    max_t = np.inf
if (max_it is np.inf) and (max_t is np.inf):
    raise ValueError('At least one stopping criterion must be specified')

assert (batch >= 1)
assert (num_workers >= 1)
assert (num_local_steps >= 1)
assert (launch_number >= 1)
assert (convergense_eps >0 )
assert (is_continue in [0,1])


def myrepr(x):
    return repr(round(x, 2)).replace('.',',') if isinstance(x, float) else repr(x)


######################
# This block varies between functions, options


def init_stepsize(P, G, X, M, p, batch, step_type):
    #TODO: init according to the theory

    #raise NotImplementedError


    if step_type == "optimal":
        la_max = scipy.linalg.eigh(a=np.multiply(P,M) @ np.linalg.inv(G) @ np.diag (p_tau**(-2)),
                                               eigvals_only=True, turbo=True, type=1, eigvals=(d - 1, d - 1))
        step_size = p_tau * np.diag(G) * la_max

        #P1 = np.diag(p**(-0.5)) @ P @ np.diag(p**(-0.5))
        #M1 = np.diag(p ** (-1)) @ M @ np.diag(p ** (-1))

        #c = scipy.linalg.eigh(a=np.multiply(P1,M1),eigvals_only=True, turbo=True, type=1, eigvals=(d - 1, d - 1))
        #step_size = (p**2) * c
        assert step_size.shape == (d,)
        return step_size

    else:
        raise ValueError("wrong step_type")


def generate_update(w, X, M, y, L, step_size, loss_func, S, probs, batch):
    assert S.shape[0] == M.shape[0]
    sample_coords = sample_coordinates(S, probs, batch)

    #print ("sample_coords: {0}".format(sample_coords))

    return w - sample_coord_grad(w, X, M, y, sample_coords, loss_func, L)/step_size

#######################


user_dir = os.path.expanduser('~/')

project_path = os.getcwd() + "/"

experiment_name = "NSYNC_" + scaled

experiment = '{0}_{1}_{2}_{3}_{4}'.format(experiment_name, loss_func, sampling_kind, step_type, batch)

logs_path = project_path + "logs{2}_{0}_{1}/".format(dataset, experiment, launch_number)
data_path = project_path + "data_{0}/".format(dataset)

if not os.path.exists(logs_path):
    os.makedirs(logs_path)

if not os.path.exists(data_path):
    os.mkdir(data_path)

data_info = np.load(data_path + 'data_info.npy')
N, L = data_info[:2]

X = np.load(data_path + 'X.npy')
y = np.load(data_path + 'y.npy')
N_X, d = X.shape

print (X.shape)
data_length_total = N_X

if not os.path.isfile(data_path + 'w_init_{0}.npy'.format(loss_func)):
    w = np.random.normal(loc=0.0, scale=1.0, size=d)
    #w = np.random.uniform(low=-100, high=100, size=d)
    np.save(data_path + 'w_init_{0}.npy'.format(loss_func), w)
    w = np.array(np.load(data_path + 'w_init_{0}.npy'.format(loss_func)))
    #print("w: {0}".format(w))
else:
    w = np.array(np.load(data_path + 'w_init_{0}.npy'.format(loss_func)))

if os.path.isfile(data_path + '{0}_f_min.npy'.format(loss_func)):
    f_min = np.float(np.load(data_path + "{0}_f_min.npy".format(loss_func)))
else:
    raise ValueError("f_min need to be founded for convergence")

#print(experiment)
#with open(logs_path + 'output' + '_' + experiment + ".txt", 'w') as f:
#with redirect_stdout(f):

currentDT = datetime.datetime.now()
print (currentDT.strftime("%Y-%m-%d %H:%M:%S"))
print (experiment)

ws = [np.copy(w)]

#grads_full = [full_grad(ws[-1], X, y,loss_func, la=L)]

S = np.arange(0, d)  # S = [n]


M = init_M (X, loss_func, L)
G = init_G (X, loss_func, dataset, L)

loss = [func(ws[-1], X, M, y,loss_func, la=L)]

probs = init_probs(d, batch, sampling_kind, M)
P = init_P_matrix(d, batch, sampling_kind, probs)

init_t_start = time.time()
if step_size is None:
    step_size = init_stepsize(P, G, X,M, probs, batch, step_type)


ts = [0]
its = [0]
it = 0

t_init = time.time() - init_t_start

t_start = time.time()
t = time.time() - t_start

#print ("it: {0} , loss: {1}".format(it, loss[-1] - f_min))
#while (it < max_it):
while (it < max_it) and (t < max_t) and (loss[-1] - f_min > convergense_eps):
#while (loss[-1] - f_min > convergense_eps):
    assert len(w) == d
    w  = generate_update(w, X, M,y, L, step_size, loss_func, S, probs, batch)

    ws.append(np.copy(w))
    loss.append(func(w, X, M, y,loss_func, la=L))
    #grads_full.append(full_grad(ws[-1], X, y, loss_func, la=L))

    t = time.time() - t_start
    ts.append(time.time() - t_start)
    it += 1
    if it % 1000 == 0:
        print(it)
    #print ("it: {0} , loss: {1}".format(it, loss[-1] - f_min))
    its.append(it)

##

#print("There were done", len(its), "iterations")
np.save(logs_path + 'loss' + '_' + experiment, np.array(loss))
np.save(logs_path + 'time' + '_' + experiment, np.array(ts))
#np.save(logs_path + 'gamma' + '_' + experiment, np.array(gammas))
#np.save(logs_path + 'grads_full' + '_' + experiment, np.array(grads_full))

#np.save(logs_path + 'information' + '_' + experiment, np.array(information_sent[::step]))
np.save(logs_path + 'iteration' + '_' + experiment, np.array(its))
np.save(logs_path + 'epochs' + '_' + experiment, np.array(its)/data_length_total)
np.save(logs_path + 'solution' + '_' + experiment, ws[-1])
#np.save(logs_path + 't-init' + '_' + experiment, t_init)

#np.save(logs_path + 'iterates' + '_' + experiment, np.array(ws))


