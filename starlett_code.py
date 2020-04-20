import math
import numpy as np
import random
import time

MIN_NUM = 1e-6
alpha = 1  # Best Result : 0.01


class Layer:
    def __init__(self, dim, unit):
        self.dim = dim
        self.unit = unit
        self.w = np.zeros((unit, dim))  # w : (unit, dim)
        self.b = np.zeros((unit, 1))  # b : (unit, 1)

    def forward(self, x):  # x : (dim, m)
        z = np.dot(self.w, x) + self.b  # z : (unit, m)
        a = sigmoid(z)  # a : (unit, m)

        a = np.minimum(1 - MIN_NUM, a)
        a = np.maximum(MIN_NUM, a)

        return a

    def backward(self, x, a, da):  # a : (unit, m)
        dz = da * derived_sigmoid(a)  # dz : (unit, m)
        dw = np.dot(dz, x.T) / x.shape[-1]   # dw : (unit, dim)
        db = np.mean(dz, axis=1, keepdims=True)  # db : (unit, 1)

        return dw, db, np.dot(self.w.T, dz)

    def update(self, dw, db):
        self.w -= alpha * dw  # w : (unit, dim)
        self.b -= alpha * db  # b : (unit, 1)

    def loss(self, y, a):
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
    x = np.random.randint(rng[0], rng[1], size=(dim, m))  # (dim, m)
    y = []
    for i in range(m):
        if x[0][i] ** 2 > x[1][i]:
            y.append(1)
        else:
            y.append(0)
    y = np.array(y).reshape((1, m))
    return x, y


# Train K times
def train(K, x, y, layers):
    start_time = time.time()

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
            dw, db, da = layer.backward(y_hats[idx + 1], y_hats[idx], da)
            layer.update(dw, db)
            idx += 1

    end_time = time.time() - start_time

    # Print cost and accuracy of train set
    # print('====================Train====================')
    print('Loss : {}'.format(layers[-1].loss(y, y_hats[0])))
    print('Accuracy : {}'.format(layers[-1].accuracy(y, y_hats[0])))
    print('Execution Time : {}'.format(end_time))


def test(y_hat, y, layers):
    start_time = time.time()

    for layer in layers:
        # Forward propagation
        y_hat = layer.forward(y_hat)

    end_time = time.time() - start_time

    # Print cost and accuarcy of test set
    # print('====================Test====================')
    print('---------------------------')
    print('Loss : {}'.format(layers[-1].loss(y, y_hat)))
    print('Accuracy : {}'.format(layers[-1].accuracy(y, y_hat)))
    print('Execution Time : {}'.format(end_time))
    # print('--------------------------------------------\n')


# Create Layers
def makeLayers(units):
    layers = []
    for i in range(len(units) - 1):
        layers.append(Layer(units[i], units[i + 1]))
    return layers


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
# task4 = makeLayers([dim, 4, 4, 4, 1])

# Train task 1
print('========== Task1 ==========')
train(K, x_train, y_train, task1)
test(x_test, y_test, task1)

# Train task 2
print('========== Task2 ==========')
train(K, x_train, y_train, task2)
test(x_test, y_test, task2)

# Train task 3
print('========== Task3 ==========')
train(K, x_train, y_train, task3)
test(x_test, y_test, task3)

# Train task 4
# print('========== Task4 ==========')
# train(K, x_train, y_train, task4)
# test(x_test, y_test, task4)