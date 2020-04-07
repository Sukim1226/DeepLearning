import math
import numpy as np
import random
import time

MIN_NUMBER = 1e-6
alpha = 0.0001    # Best Result : 0.01

class Network:
    def __init__(self, dim, unit):
        self.dim = dim
        self.unit = unit
        self.w = np.zeros((unit, dim)) # w : (unit, dim)
        self.b = np.zeros((unit, 1)) # b : (unit, 1)

    def forward(self, x): # x : (dim, m)
        z = np.dot(self.w, x) + self.b # z : (unit, m)
        a = sigmoid(z) # a : (unit, m)

        a = np.minimum(1 - MIN_NUMBER, a)
        a = np.maximum(MIN_NUMBER, a) 

        return a
        
    def backward(self, x, y, a, da): # y : (1, m) / a : (unit, m)
        dz = da * derived_sigmoid(a)
        dw = np.mean(np.dot(dz, x.T), axis=1, keepdims=True) # dw : (unit, dim)
        db = np.mean(dz, axis = 1, keepdims=True) # db : (unit, 1)
        
        return dw, db, np.dot(self.w.T, dz)

    def update(self, dw, db):
        self.w -= alpha * dw # w : (unit, dim)
        self.b -= alpha * db # b : (unit, 1)

    def cost(self, y, a):
        c = -np.mean(y * np.log(a) + (1 - y) * (np.log(1 - a)))
        return c

    def accuracy(self, y, a):
        a = np.round(a)
        bool_arr = ~np.logical_xor(a, y)
        return np.mean(bool_arr)
                    

# Activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derived_sigmoid(a):
    return a * (1 - a)


# Generate train sample shaped (dim, m)
def generate(dim, m, rng): 
    x = np.random.randint(rng[0], rng[1], size=(dim, m)) # (dim, m)
    y = [] 
    for i in np.sum(x, axis=0):
        if i > 0:
            y.append(1)
        else:
            y.append(0)
    y = np.array(y) # (1, m)
    return x, y


# Train K times
def train(K, x, y, layers):
    for i in range(K):
        y_hats = [x]

        # Forward propagation
        for layer in layers:
            y_hat = layer.forward(y_hats[-1])
            y_hats.append(y_hat)

        y_hats.reverse()
        idx = 0
        da = (-y / y_hats[idx]) + (1 - y) / (1 - y_hats[idx])

        # Back propagation
        for layer in reversed(layers):
            dw, db, da = layer.backward(y_hats[idx + 1], y, y_hats[idx], da)
            layer.update(dw, db)
            idx += 1

    # Print cost and accuracy
    print(layers[-1].cost(y, y_hats[0]))
    print(layers[-1].accuracy(y, y_hats[0]))


# Create Layers
def makeLayers(units):
    Layers = []
    for i in range (len(units) - 1):
        Layers.append(Network(units[i], units[i + 1]))
    return Layers


# Initialize input dimension, # train set, # test set, random data range, iteration
dim = 2
m = 1000
n = 100
data_range = [-2, 2]
K = 1000

# Generate train samples
x_train, y_train = generate(dim, m, data_range)
x_test, y_test = generate(dim, n, data_range)

# Number of units in each layer (counting from input)
task1 = makeLayers([dim, 1])
task2 = makeLayers([dim, 1, 1])
task3 = makeLayers([dim, 3, 1])

# Train task 1
print('========== Task1 ==========')
train(K, x_train, y_train, task1)
train(K, x_test, y_test, task1)

# Train task 2
print('========== Task2 ==========')
train(K, x_train, y_train, task2)
train(K, x_test, y_test, task2)

# Train task 3
print('========== Task3 ==========')
train(K, x_train, y_train, task3)
train(K, x_test, y_test, task3)