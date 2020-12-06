#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DLA1: Question 1 --- The Perceptron
This is a Python script for running a simple perceptron classifying an 
input vector x of 10 binary (+/- 1) inputs. It classifies the sum of the 
elements (x_i) to be "positive" or "negative".

@author: benjamintenmann
"""
import numpy as np
import random
from scipy import stats
from math import exp
import multiprocessing as mp
import matplotlib.pyplot as plt
import time 



def fMSE(prediction, real):
    if type(prediction) == float:
        return (prediction-real)**2
    else:
        SE = list(map(lambda x, y: (x-y)**2, prediction, real))
        MSE = sum(SE)/len(SE)
        return MSE


def compute(x, w, i, step):
    logistic = lambda r : 1/(1+exp(-r))
    d_logistic = lambda x_i, w_i : (x_i * exp(-(x_i * w_i)))/(1-exp(-(x_i * w_i)))**2
    gradient_descent = lambda step, t, y, g: (1/step) * (t-y) * g
    
    y = logistic(x[:-1].dot(w))
    t = x[-1]
    g = d_logistic(x[i], w[i])
    
    w[i] = w[i] + gradient_descent(step, t, y, g)
    return w[i]


def epoch(mtX, w):
    """
    Parameters
    ----------
    mtX : The train dataset matrix subset.
    w : The model weights vector.

    Returns
    -------
    w : Updated weights vector.

    """
    print('epoch commences...')
    start = time.time()
    X = mtX.copy()
    np.random.shuffle(X)
    step = 1
    ax = plt.subplot(1,1,1)
    prediction = 1/(1+exp(-(X[0, :-1].dot(w))))
    real = X[0,-1]
    ax.scatter(step, fMSE(prediction, real))
    for x in X:
        args = [*zip([x]*10, [w]*10, list(range(10)), [step]*10)]
        pool = mp.Pool(10)
        w = np.array(list(pool.starmap(compute, args)))
        pool.close()
        step += 1
        prediction = 1/(1+exp(-(x[:-1].dot(w))))
        real = x[-1]
        ax.plot(step, fMSE(prediction, real))
    end = time.time()
    print('completed! t:', end-start,'\n')
    return w
    

class perceptron:
    """
    The perceptron: a simple a neuron taking an input vector of ten length = 10.
    Each element in the vector is either -1 or 1, except the last value is either 
    0 (sum of preceding values are <0) or 1 (preceding values sum to >= 0).
    
    The perceptron is an object-class and can be initialised with a 2D array of input 
    row vectors. 
    """
    
    def __init__(self, A):
        self.data = A
        
    def learn(self, inds, epochs = 1000):
        print('neuron starts learning...\n\n')
        B = self.data
        X = B[inds, :]
        
        W = w_n = np.ones(10)
        n_epochs = range(epochs)
        for ep in n_epochs:
            w_n = epoch(X, w_n)
            W = np.vstack([W, w_n])
        print('done!\n\n')
        return W # return weight matrix --- rows are different epochs
        
    def test(self, inds, w):
        print('neuron starts testing...\n')
        B = self.data
        r_ind = np.arange(1, B.shape[0])
        ind = r_ind[np.in1d(r_ind, inds, invert=True)]
        Y = B[ind, :].copy()
        prediction = list(map(lambda x, w : 1/(1+exp(-(x.dot(w)))), Y[:,:-1], [w for x in range(Y.shape[0])]))
        real = Y[:, -1]
        print('prediction:', prediction[0:10])
        print('real:', real[0:10])
        MSE = fMSE(prediction, real)
        print('done!')
        return MSE
        
if __name__ == '__main__':    
    # creating random data
    n = 1000000
    B = np.random.randint(0, 2, size=(n,10))
    B = np.where(B == 0, -1, B)
    A = np.zeros((n, 11))
    A[:, :-1] = B
    A[:,10] = [1 if sum(A[i,:]) >= 0 else 0 for i in range(A.shape[0])]
    inds = np.random.randint(0, A.shape[0], size=int(A.shape[0] * 0.7))
    
    # initialise perceptron object
    neuron = perceptron(A)
    weights = neuron.learn(inds, epochs=10)
    errors = [neuron.test(inds, m) for m in weights]
    eps = list(range(1, 12))
    
    plt.scatter(eps, errors)
    plt.show()



