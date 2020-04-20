import numpy as np
import time


alpha = 0.01
w = np.zeros((1, 2))
b = np.zeros((1, 1))


def generate(input_dimension, sample_num):
    x = np.random.uniform(-2, 2, size=(input_dimension, sample_num))  # x_train : (2, 1000) / x_test : (2, 100)
    y = []
    for i in range(sample_num):
        if x[0][i] * x[0][i] > x[1][i]:
            y.append(1)
        else:
            y.append(0)
    y = np.array(y)
    return x, y


def forward(x):
    global w, b
    z = np.dot(w, x) + b
    a = 1 / (1 + np.exp(-z))

    return a


def backward(x, y, a):
    global w, b
    dz = a - y
    # dw = np.mean(np.dot(dz, x.T), axis=1, keepdims=True)
    # db = np.mean(dz, axis=1, keepdims=True)
    dw = 1/(np.size(x, axis=1)) * np.sum(np.dot(dz, x.T))
    db = 1/(np.size(x, axis=1)) * np.sum(np.dot(dz))

    w -= alpha * dw
    b -= alpha * db


def accuracy(y, y_hat):
    y_hat = np.round(y_hat)
    return np.mean(~np.logical_xor(y_hat, y))


def train(iteration, x, y):
    for i in range(iteration):
        a = forward(x)
        backward(x, y, a)

    print('[Task 1] Train Accuracy: {}'.format(accuracy(y, a)))


def test(x, y):
    a = forward(x)
    print('[Task 1] Test Accuracy: {}'.format(accuracy(y, a)))


x_train, y_train = generate(2, 1000)
x_test, y_test = generate(2, 1000)


if __name__ == '__main__':
    train(100, x_train, y_train)
    test(x_test, y_test)