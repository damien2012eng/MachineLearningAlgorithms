# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:31:07 2019

@author: damien
"""
import numpy as np
import matplotlib.pyplot as plt

def random_func(x):
    return x[0]**2+x[1]+5*x[2]

def random_func_1(x):
    return x[0]**2+0.5*x[1]+5*x[2]

def generate_data(start, end, num):
    data = []
    iteration = np.linspace(start, end, num)
    for i in iteration:
        for j in iteration:
            for o in iteration:
                data.append([i,j,o])
    return np.array(data)/10

def calc_y(func, data):
    return np.apply_along_axis(random_func, 1, data)/10

#Training datasets
training_set = generate_data(1,7,7)
training_true = calc_y(random_func, training_set)

#Testing datasets
testing_set = generate_data(1.2,6,5)
testing_true = calc_y(random_func, testing_set)
