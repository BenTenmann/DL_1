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
import math
import multiprocessing as mp
import matplotlib.pyplot as plt

class perceptron(A):
    """
    The perceptron: a simple a neuron taking an input vector of ten length = 10.
    Each element in the vector is either -1 or 1, except the last value is either 
    0 (sum of preceding values are <0) or 1 (preceding values sum to >= 0).
    
    The perceptron is an object-class and can be initialised with a 2D array of input 
    row vectors. 
    """
    
    def __init__(self, A):
        self.train_test = A[np.random.randint(0, A.shape[0]+1, size=int(A.shape[0] * 0.7)), :]
        
    def train():
        w = np.ones(10)
        X = self.train_test
        exp = math.exp
        
        logistic = lambda r : 1/(1+exp(-r))
        d_logistic = lambda x_i, w_i : (x_i * exp(-(x_i * w_i)))/(1-exp(-(x_i * w_i)))**2
        gradient_descent = lambda step, t, y, g: (1/step) * (t-y) * g
        step = 1
        for x in X:
            for i, x_i in enumerate(x[:-1]):
                y = logistic(x[:-1].dot(w))
                t = x[-1]
                g = d_logistic(x_i, w[i])
                
                w[i] = w[i] - gradient_descent(step, t, y, g) 
                
            step += 1
        
    
    def test():
        

# creating random data
n = 1000000
B = np.random.randint(0, 2, size=(n,10))
B = np.where(B == 0, -1, B)
A = np.zeros((n, 11))
A[:, :-1] = B
A[:,10] = [1 if sum(A[i,:]) >= 0 else 0 for i in range(A.shape[0])]


plt.scatter(res_x, res_y)
plt.show()

# initialise perceptron object
neuron = perceptron(A)



import time 

start = time.time()

end = time.time()
print(end - start)

start = time.time()

end = time.time()
print(end - start)

