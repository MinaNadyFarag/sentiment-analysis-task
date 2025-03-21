# Copyright (c) 2025 [MinaNadyFarag]. All rights reserved.
# This code is the property of [MinaNAdyFarag] and may not be copied, distributed, or used without permission.
# model_architecture.py

import numpy as np

def relu(z):
    result = z.copy()
    result[result < 0] = 0
    return result

def softmax(z):
    e_z = np.exp(z)
    sum_e_z = np.sum(e_z)
    return e_z / sum_e_z

def forward_propagation(x, W1, W2, b1, b2):
    z1 = np.dot(W1, x) + b1
    h = relu(z1)
    z2 = np.dot(W2, h) + b2
    y_hat = softmax(z2)
    return y_hat, h