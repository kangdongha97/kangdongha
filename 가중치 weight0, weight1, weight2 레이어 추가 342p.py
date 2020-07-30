# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:03:41 2020

@author: 동211_로봇SI-02
"""


import numpy as np

def actf(x):
    return 1/(1+np.exp(-x))

def actf_deriv(x):
    return x*(1-x)

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])

y = np.array([[0],[1],[1],[0]])

inputs = 3
hiddens = 6
hiddens1 = 8
outputs = 1

weight0 = 2*np.random.random((inputs, hiddens))-1
weight1 = 2*np.random.random((hiddens, hiddens1))-1
weight2 = 2*np.random.random((hiddens1, outputs))-1

for i in range(10000):
    layer0 = X
    
    net1 = np.dot(layer0, weight0)
    layer1 = actf(net1)
    layer1[:, -1] = 1.0
    
    net2 = np.dot(layer1, weight1)
    layer2 = actf(net2)
    layer2[:, -1] = 1.0
    
    net3 = np.dot(layer2, weight2)
    layer3 = actf(net3)
    
    
    layer3_error = layer3-y
    
    layer3_delta = layer3_error*actf_deriv(layer3)
    
    layer2_error = np.dot(layer3_delta, weight2.T)
    
    layer2_delta = layer2_error*actf_deriv(layer2)
    
    layer1_error = np.dot(layer2_delta, weight1.T)
    
    layer1_delta = layer1_error*actf_deriv(layer1)
    

    
    weight2 += -0.2*np.dot(layer2.T, layer3_delta)
    
    weight1 += -0.2*np.dot(layer1.T, layer2_delta)
    
    weight0 += -0.2*np.dot(layer0.T, layer1_delta)
    
print(" layer2\n", layer2, "\n")
print(" layer3\n", layer3, "\n")
print(" weight0\n", weight0, "\n")
print(" weight1\n", weight1, "\n")
print(" weight2\n", weight2, "\n")
print(" x\n", X, "\n")
print(" y\n", y, "\n")