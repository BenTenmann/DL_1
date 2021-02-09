#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:42:19 2020

@author: benjamintenmann
"""

import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt

def g(x):
    z = 1/(1+np.exp(-x))
    return z

def gprime(x):
    der = g(x) * (1-g(x))
    return der

def loss(t, y):
    l = 0.5 * sum((t-y)**2)
    return l

bias = -1
epsilon = 2.3
alpha = 0.001

targets = np.identity(8)
inputs = np.vstack((targets, (np.ones(8)*bias)))

I = 8
J = 3
K = 8

epochs = 1000

W1 = uniform.rvs(size=(J*(I+1))).reshape(J, (I+1))
W2 = uniform.rvs(size=(K*(J+1))).reshape(K, (J+1))

delta_j = np.zeros(J)

epoch_error = []

for i in range(epochs):
    DW1 = np.zeros((J, I+1))
    DW2 = np.zeros((K, J+1))
    
    ep_er = 0
    for n in range(inputs.shape[1]):
        
        z_i = inputs[:, n]
        t_k = targets[:, n]
        
        x_j = W1 @ z_i
        z_j = [g(q) for q in x_j]
        z_j.append(bias)
        
        x_k = W2 @ z_j
        z_k = g(x_k)
        
        ep_er += loss(t_k, z_k)
        delta_k = gprime(x_k) * (t_k - z_k)
        
        DW2 = ((1+alpha) * DW2) + np.outer(delta_k, z_j)
        
        delta_j = gprime(x_j) * (delta_k @ W2[:,:-1])
        
        DW1 = ((1+alpha) * DW1) + np.outer(delta_j, z_i)
    epoch_error.append(ep_er)
    W2 += epsilon * DW2
    W1 += epsilon * DW1


plt.plot(list(range(1,epochs+1)), epoch_error)
plt.show()

import pandas as pd
df = pd.DataFrame({"epoch":list(range(1, epochs+1)), "error":epoch_error})
df.to_csv("/Users/benjamintenmann/Desktop/CompBio/Assignments/DL_1/MLP/alpha.csv", index=False)



