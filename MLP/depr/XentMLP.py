#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 18:20:17 2020

@author: benjamintenmann
"""


import numpy as np
from math import log
from scipy.stats import uniform
import matplotlib.pyplot as plt

def g(x):
    """
    

    Parameters
    ----------
    x : np.array1d
        Vector result of the weight matrix - input vector dot product.

    Returns
    -------
    z : np.array1d
        The vector containing the activation of each unit. The ativation is sigmoidal.

    """
    z = 1/(1+np.exp(-x))
    return z

def gprime(x):
    """
    

    Parameters
    ----------
    x : np.array1d
        Vector containing the weighted-sums of the inputs.

    Returns
    -------
    der : np.array1d
        Vector containing the derivative of the activation function.

    """
    der = g(x) * (1-g(x))
    return der

def softmax(z):
    """
    

    Parameters
    ----------
    z : np.array1d
        Vector containing the unit activations.

    Returns
    -------
    S : np.array1d
        Vector with the discrete probability distribution --- softmax-transformed output.

    """
    S = np.exp(z)/np.sum(np.exp(z))
    return S

def CrossEntropy(t, S):
    """
    

    Parameters
    ----------
    t : np.array1d
        Target vector --- i.e. the training signal.
    S : np.array1d
        The softmax transformed output vector.

    Returns
    -------
    float
        A scalar measuring the cross-entropy between the target and output distributions.

    """
    
    X = t * np.log(S)
    
    return -np.sum(X)

def d_S(S_i, S_j):
    """
    

    Parameters
    ----------
    z : np.array1d
        Vector containing the activations of the output layer.

    Returns
    -------
    DS : np.array2d
        A n x n Jacobian matrix of the derivatives of each final output with respect to one of the inputs.
        Rows are the different different outputs and columns are the different inputs.

    """        
    if S_i == S_j:
        delta = 1
    else:
        delta = 0
            
    DS = S_i * (delta - S_j)
            
    return DS

compute_jacob = np.frompyfunc(d_S, 2, 1)

def d_Xent(t, S):
    """
    

    Parameters
    ----------
    t : np.array1d
        1-HOT encoded class vector.
    S : np.array1d
        The softmax-transformed output unit actications.

    Returns
    -------
    np.array1d
        The derivative of the softmax function.

    """
    
    d_X = t / s
    
    return d_X

bias = -1
epsilon = 2.3

targets = np.identity(8) # data --- 8 classes; 1-hot encoded
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
        z_j = np.append(g(x_j), bias)
        
        x_k = W2 @ z_j
        z_k = g(x_k)
        
        s = softmax(z_k)
        
        ep_er += CrossEntropy(t_k, s)
        
        DS = compute_jacob.outer(s, s).astype(np.float64)
        delta_k = d_Xent(t_k, s) @ (DS * gprime(x_k)) #fix this
        
        DW2 = DW2 + np.outer(delta_k, z_j)
        
        delta_j = gprime(x_j) * (delta_k @ W2[:,:-1])
        
        DW1 = DW1 + np.outer(delta_j, z_i)
    epoch_error.append(ep_er.copy())
    W2 += epsilon * DW2
    W1 += epsilon * DW1


plt.plot(list(range(1,epochs+1)), epoch_error)
plt.show()


