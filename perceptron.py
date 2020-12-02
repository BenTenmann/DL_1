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

class perceptron(A):
    """
    The perceptron: a simple a neuron taking an input vector of ten length = 10.
    Each element in the vector is either -1 or 1, except the last value is either 
    0 (sum of preceding values are <0) or 1 (preceding values sum to >= 0).
    
    The perceptron is an object-class and can be initialised with a 2D array of input 
    row vectors. 
    """
    
    def __init__(self, A):
        self.train = A[np.random.randint(0, A.shape[0]+1, size=int(A.shape[1] * 0.7)), :]
        
    def train_net():
        
        
    
    def test_net():
        
    
# creating random data
n = 10
B = np.random.randint(0, 2, size=(n,10))
B = np.where(B == 0, -1, B)
A = np.zeros((n, 11))
A[:, :-1] = B
A[:,10] = [1 if sum(A[i,:]) >= 0 else 0 for i in range(A.shape[0])]

# initialise perceptron object
neuron = perceptron(A)


