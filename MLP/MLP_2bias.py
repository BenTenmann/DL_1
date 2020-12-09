#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP: Multi-layer perceptron
A MLP to solve non-linear problems

@author: benjamintenmann
"""

import numpy as np
import matplotlib.pyplot as plt
import math

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
    error = list(map(err, Y[:, :-1], y_hat))
    MSE = sum(error)/len(error)
    return MSE

def delta(z, error):
    derivative = d_activation(z)
    return derivative * error

def forprop(x, *M):
    z = x
    res = []
    for i, W in enumerate(M):
        if i == 1:
            z = np.append(z, -1)
        z = activation(weighted_sum(z, W))
        res.append(z)
    return res

def backprop(x, step, M):
    z_j, z_k = forprop(x, M[0], M[1])
    error = x[:-1] - z_k
    z_j = np.append(z_j, -1)
    delta_k = delta(weighted_sum(z_j, M[1]), error)
    W_2 = M[1] + ((1/step) * np.outer(z_j, delta_k))
    
    delta_j = delta(weighted_sum(x, M[0]), weighted_sum(delta_k, np.delete(W_2, 3, axis=0).T))
    W_1 = M[0] + ((1/step) * np.outer(x, delta_j))
    
    return W_1, W_2

def epoch(mtX, M):
    X = mtX.copy()
    np.random.shuffle(X)
    
    step = 10
    for x in X:
        M = backprop(x, step, M)
        step += 10
    return M

class MLP:
    
    def __init__(self, wshape = [(9,3), (4,8)]):
        self.data = np.identity(8)
        self.weights = [np.ones(m) for m in wshape]
        
    def train(self, inds, epochs=1000):
        print('neural net begins to train...\n')
        B = np.ones((8,9)) * -1
        B[:, :-1] = self.data
        X = B[inds, :]
        M = self.weights
        ls = [M]
        print('progress:')
        for r in range(epochs):
            M = epoch(X, M)
            ls.append(M)
            printDone(r, epochs)
        print('\ndone!\n\n')
        return ls
    
    def test(self, inds, M):
        print('testing epochs...')
        B = np.ones((8,9)) * -1
        B[:, :-1] = self.data
        r_ind = np.arange(0, 8)
        ind = r_ind[np.in1d(r_ind, inds, invert=True)]
        Y = B[ind, :]
        errs = [E_r(Y, m) for m in M]
        print('\ndone!\n\n')
        return errs
        

        
        
if __name__=='__main__':
    n = 100
    neural_net = MLP()
    inds = np.random.choice(np.arange(0,8), size=6, replace=False)
    M = neural_net.train(inds, epochs = n)
    errors = neural_net.test(inds, M)
    eps = list(range(n+1))
    
    plt.plot(eps, errors)
    plt.show()
    
    
    
    