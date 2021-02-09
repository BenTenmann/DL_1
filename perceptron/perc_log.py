#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 20:21:59 2020

@author: benjamintenmann
"""


import numpy as np
from itertools import product
from math import exp
from scipy.stats import uniform
import matplotlib.pyplot as plt

def g(x):
    z = 1/(1+exp(-x))
    return z

def gprime(x):
    der = g(x) * (1-g(x))
    return der

def loss(t,y):
    err = 0.5 * (t-y)**2
    return err

def ifelse(condition, t=1, f=0):
    if condition == True:
        return t
    else:
        return f

def gradient_descent(t, w, x):
    z = x.dot(w)
    delta_w = (t-g(z)) * gprime(z) * x
    return delta_w

epochs = 200
epsilon = 0.01
inputs = list(product([-1,1], repeat=10))

w = uniform.rvs(size=11)


# training
ep_err = []
W = [w.copy()]
for i in range(epochs):
    
    err = 0
    
    for n in range(819):
        
        z = np.array(inputs[n])
        t = ifelse(sum(z) >= 0)
        
        x = np.append(z, -1)
        
        
        w += epsilon * gradient_descent(t, w, x)
        err += loss(t, g(x.dot(w)))
        
        
    W.append(w.copy())
    ep_err.append(err/819)

plt.plot(list(range(epochs)), ep_err)
plt.show()

# testing
test_err = []
for w in W:
    
    err = 0
    
    for p in range(205):
        
        z = np.array(inputs[p+819])
        t = ifelse(sum(z) >= 0)
        
        x = np.append(z, -1)
        
        err += loss(t, g(x.dot(w)))
    
    test_err.append(err/205)
    
plt.plot(list(range(epochs+1)), test_err)
plt.show()

import pandas as pd
T = pd.DataFrame({"err_train":ep_err, "err_test":test_err[:-1]})
T.to_csv("/Users/benjamintenmann/Desktop/CompBio/Assignments/DL_1/perceptron/perc_out.csv")

