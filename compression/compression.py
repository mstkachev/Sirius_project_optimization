import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.sgd import SGD
from torch.utils.data import TensorDataset, DataLoader, Dataset,Subset
from torchvision import datasets, transforms, models
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib
import collections
from collections import defaultdict
from sklearn import datasets
from sklearn.datasets import load_digits
from functools import reduce

from torch.optim import SGD


############## Compressed SGD/GD #####################

class CSGD(SGD):
    def __init__(self, params, lr, comp=None, filename = 1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super(CSGD, self).__init__(params, lr, momentum, dampening,
                 weight_decay, nesterov)
        self.comp = comp
        self.filename = filename
    
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                if self.comp == None:
                    d_p = p.grad.data
                else:
                    b = (p.grad.data).reshape(-1).to('cpu').numpy()
                    if self.comp == 'C_10':
                        d_p = C_10(p.grad.data)
                    if self.comp == 'C_2':
                        d_p = C_2(p.grad.data)
                    if self.comp == 'C_2_d':
                        d_p = C_2_d(p.grad.data)    
                    if self.comp == 'T':
                        d_p = topk(p.grad.data, len(b) // 5)
                    if self.comp == 'T_25':
                        try:
                            d_p = topk(p.grad.data, 25)
                        except:
                            d_p = p.grad.data
                    if self.comp == 'T_C_10':
                        d_p = C_10(topk(p.grad.data, len(b) // 4))
                    if self.comp == 'T_C_2':
                        d_p = C_2(topk(p.grad.data, len(b) // 4))
                    if self.comp == 'R_3':
                        d_p = randomk(p.grad.data, 3)
                    if self.comp == 'R':
                        d_p = randomk(p.grad.data, len(b) // 5)
                    if self.comp == 'R_v2':
                        d_p = randomk_v2(p.grad.data, len(b) // 5)

                                       
                    if self.filename != None:
                        a = d_p.reshape(-1).to('cpu').numpy()
                        with open('/content/gdrive/My Drive/KAUST/ResNet.csv', "ab") as f:
                            c = np.linalg.norm(a-b) ** 2
                            d = np.linalg.norm(b) ** 2
                            np.savetxt(f, np.array([c / d]))
                
                
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
        
        
 ###### Compressions Examples ############

# Compressions with Pytorch library


def get_device(x):
    if x.is_cuda:
        device='cuda'
    else:
        device='cpu'
    return device

def top1(x):
    device = get_device(x)
    dim = x.shape
    m = torch.max(torch.abs(x))
    k = torch.zeros(dim).to(device)
    ans = torch.where(torch.abs(x) >= m, x, k)
    return ans

def topk(x, k):
    device = get_device(x)
    dim=x.shape
    h = x.reshape(-1)
    sor,_ = torch.sort(abs(h))
    m = sor[-k]
    z = torch.zeros(dim).to(device)
    ans = torch.where(torch.abs(x) >= m, x, z)
    return ans

def randomk(x, k):
    device = get_device(x)
    dim=x.shape
    n = (x.reshape(-1)).shape[0]
    p = k / n
    pr = (torch.ones(dim) * p).to(device)
    prob = torch.distributions.bernoulli.Bernoulli(pr)
    return (x * prob.sample()) / p

def randomk_v2(x, k):
    device = get_device(x)
    dim=x.shape
    n = (x.reshape(-1)).shape[0]
    p = k / n
    pr = (torch.ones(dim) * p).to(device)
    prob = torch.distributions.bernoulli.Bernoulli(pr)
    return (x * prob.sample()) 
                            


def C_10(x):
    dim = x.shape
    device = get_device(x)
    s = torch.floor(torch.where(x!=0, torch.log10(torch.abs(x)), x))
    po = torch.pow(10,s)
    c = abs(x) // po
    s = torch.where(c == 1, s - 1, s)
    po = torch.pow(10,s)
    c = abs(x) // po
    z = (torch.ones(dim) * 15).to(device)
    c = torch.where((c > 15) & (c < 20), z, c)
    ro = c * po
    p = abs(abs(x) - ro) / po
    p = torch.where(c==15, p / 5, p)
    prob = torch.zeros(dim)
    try:
        prob = torch.distributions.bernoulli.Bernoulli(p).sample()
        prob = torch.where(c==15, prob * 5, prob)
    except:
        pass
    ans = torch.sign(x) * (abs(c) + prob) * po
    return ans.to(device)


def C_2(x):
    dim = x.shape
    device = get_device(x)
    s = torch.floor(torch.where(x!=0, torch.log2(torch.abs(x)), x))
    po = torch.pow(2,s)
    p = abs(abs(x) - po) / po
    prob = torch.zeros(dim)
    try:
        prob = torch.distributions.bernoulli.Bernoulli(p)
    except:
        pass
    ans = torch.sign(x) * (1 + prob.sample()) * po
    return ans.to(device)

def C_2_d(x):
    dim = x.shape
    device = get_device(x)
    s = torch.floor(torch.where(x!=0, torch.log2(torch.abs(x)), x))
    po = torch.pow(2,s)
    p = abs(abs(x) - po) / po
    prob = torch.zeros(dim).to(device)
    z = torch.ones(dim).to(device)
    prob = torch.floor(torch.where(p > 1/2, z, prob))
    ans = torch.sign(x) * (1 + prob.to(device)) * po
    return ans.to(device)
