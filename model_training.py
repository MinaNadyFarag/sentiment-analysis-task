# Copyright (c) 2025 [MinaNadyFarag]. All rights reserved.
# This code is the property of [MinaNAdyFarag] and may not be copied, distributed, or used without permission.
# model_training.py

import numpy as np
from model_architecture import forward_propagation, relu

def cross_entropy_loss(y_hat, y):
    return -np.sum(y * np.log(y_hat))

def backpropagation(x, y, y_hat, h, W1, W2, b1, b2):
    grad_b2 = y_hat - y
    grad_W2 = np.dot(grad_b2, h.T)
    grad_b1 = relu(np.dot(W2.T, grad_b2))
    grad_W1 = np.dot(grad_b1, x.T)
    return grad_W1, grad_W2, grad_b1, grad_b2

def gradient_descent(W1, W2, b1, b2, grad_W1, grad_W2, grad_b1, grad_b2, alpha):
    W1_new = W1 - alpha * grad_W1
    W2_new = W2 - alpha * grad_W2
    b1_new = b1 - alpha * grad_b1
    b2_new = b2 - alpha * grad_b2
    return W1_new, W2_new, b1_new, b2_new

def train_model(words, word2Ind, V, epochs=100, alpha=0.03):
    N = 3  # Size of the embedding vector
    W1 = np.random.rand(N, V)
    W2 = np.random.rand(V, N)
    b1 = np.random.rand(N, 1)
    b2 = np.random.rand(V, 1)

    for epoch in range(epochs):
        for x_array, y_array in get_training_example(words, 2, word2Ind, V):
            x = x_array.reshape(V, 1)
            y = y_array.reshape(V, 1)
            y_hat, h = forward_propagation(x, W1, W2, b1, b2)
            loss = cross_entropy_loss(y_hat, y)
            grad_W1, grad_W2, grad_b1, grad_b2 = backpropagation(x, y, y_hat, h, W1, W2, b1, b2)
            W1, W2, b1, b2 = gradient_descent(W1, W2, b1, b2, grad_W1, grad_W2, grad_b1, grad_b2, alpha)
        print(f"Epoch {epoch+1}, Loss: {loss}")
    
    return W1, W2, b1, b2