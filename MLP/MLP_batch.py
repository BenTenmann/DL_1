#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP: Multi-layer perceptron
A MLP to solve non-linear problems

@author: benjamintenmann
"""

import numpy as np
import math
from scipy.stats import uniform
import matplotlib.pyplot as plt

def printDone(i, ep):
    if (i+1) % (ep*0.1) == 0:
        frac = int((i+1)/(ep*0.1))
        print('[*'+('*'*frac)+(' '*(10-frac))+']')

def weighted_sum(x, W):
    return x.dot(W)

def activation(z):
    exp = math.exp
    y = 1/(1+np.array([exp(-n) for n in z]))
    return y

def d_activation(z):
    exp = math.exp
    derivative = np.array([exp(-m) for m in z])/(1+np.array([exp(-n) for n in z]))**2
    return derivative

def E_r(Y, M):
    y_hat = np.array([forprop(y, M[0], M[1])[1] for y in Y])
    print('prediction:', y_hat, '\nreal:', Y, '\n\n', sep='\n')
    err = lambda t, y: 0.5 * sum((t-y)**2)
    error = list(map(err, Y, y_hat))
    MSE = sum(error)
    return MSE

def delta(z, error):
    derivative = d_activation(z)
    return derivative * error

def forprop(x, *M):
    z = x
    res = []
    for W in M:
        z = activation(weighted_sum(z, W))
        res.append(z)
    return res
    
def backprop(x, M, W):
    z_j, z_k = forprop(x, M[0], M[1])
    error = x - z_k
    delta_k = delta(weighted_sum(z_j, M[1]), error)
    W_2 = W[1] + np.outer(z_j, delta_k)
    delta_j = delta(weighted_sum(x, M[0]), weighted_sum(delta_k, W_2.T))
    W_1 = W[0] +  np.outer(x, delta_j)
    return W_1, W_2

def batch(W, M, step):
    W_1 = M[0] + ((1/step) * W[0])
    W_2 = M[1] + ((1/step) * W[1])
    return W_1, W_2

def epoch(mtX, M, step):
    X = mtX.copy()
    np.random.shuffle(X)
    
    W = (np.zeros((8,3)), np.zeros((3,8)))
    for x in X:
        W = backprop(x, M, W)
    M = batch(W, M, step)
    return M

class MLP:
    
    def __init__(self, wshape = [(8,3), (3,8)]):
        self.data = np.identity(8)
        self.weights = [uniform.rvs(size=24).reshape(m) for m in wshape]
        
    def train(self, inds, epochs=1000):
        print('neural net begins to train...\n')
        B = self.data
        X = B[inds, :]
        M = self.weights
        ls = [M]
        print('progress:')
        step = 1
        for r in range(epochs):
            M = epoch(X, M, step)
            ls.append(M)
            step += 1
            printDone(r, epochs)
        print('\ndone!\n\n')
        return ls
    
    def test(self, inds, M):
        print('testing epochs...')
        B = self.data
        r_ind = np.arange(0, 8)
        ind = r_ind[np.in1d(r_ind, inds, invert=True)]
        Y = B[np.arange(0,8), :]
        errs = [E_r(Y, m) for m in M]
        print('\ndone!\n\n')
        return errs
        

        
        
if __name__=='__main__':
    n = 15000
    neural_net = MLP()
    inds = np.random.choice(np.arange(0,8), size=6, replace=False)
    M = neural_net.train(np.arange(0,8), epochs = n)
    errors = neural_net.test(inds, M)
    eps = list(range(n+1))
    
    plt.plot(eps, errors)
    plt.show()
    
    
    
    