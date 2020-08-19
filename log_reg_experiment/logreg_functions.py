import numpy as np
import random
import time

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize

supported_penalties = ['l1', 'l2']

def logreg_loss(w, X, y, la):
    assert la >= 0
    assert len(y) == X.shape[0]
    assert len(w) == X.shape[1]
    l = np.log(1 + np.exp(-X.dot(w) * y))
    m = y.shape[0]
    return np.mean(l) + la * regularizer(w)

def logreg_grad(w, X, y, la):
    """
    Returns full gradient
    :param w:
    :param X:
    :param y:
    :param la:
    :return:
    """
    assert la >= 0
    assert (len(y) == X.shape[0])
    assert (len(w) == X.shape[1])
    loss_grad = np.mean([logreg_sgrad(w, X[i], y[i], la) for i in range(len(y))], axis=0)
    assert len(loss_grad) == len(w)
    return loss_grad + la * regularizer_grad(w)

def logreg_sgrad(w, x_i, y_i, la):
    """
    Returns one stochastic gradient
    :param w: target variable
    :param x_i: i-th row if the data matrix, i.e. A[i,:]
    :param y_i: i-th label corresponding to the i-th row if the data matrix
    :param la: regularization parameter
    :return: one stochastic gradient
    """
    assert (la >= 0)
    assert (len(w) == len(x_i))
    assert y_i in [-1, 1]
    loss_sgrad = - y_i * x_i / (1 + np.exp(y_i * np.dot(x_i, w)))
    assert len(loss_sgrad) == len(w)
    return loss_sgrad

def regularizer(w: np.ndarray):
    return np.sum(w**2/(1 + w**2))

def regularizer_grad(w):
    return 2*w /(1 + w**2)**2

def sample_logreg_sgrad(w, X, y, la, batch):
    """
    Returns minibatch stochastic gradient
    :param w: target variable
    :param X: data matrix
    :param y: label column
    :param la: regularization parameter
    :param batch: batch_size
    :return:
    """
    assert la >= 0
    n, d = X.shape
    assert(len(w) == d)
    assert(len(y) == n)
    grad_sum = np.zeros(d)
    i_batch = np.random.choice(n, batch)
    assert(i_batch == batch)
    return np.sum(logreg_sgrad(w, X[i_batch], y[i_batch], la))/batch + la * regularizer_grad(w)
