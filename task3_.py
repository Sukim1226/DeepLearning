import numpy as np
import time
import task1


w1 = np.zeros((3, 2))
b1 = np.zeros((3, 1))
w2 = np.zeros((1, 3))
b2 = np.zeros((1, 1))


def forward(x):
    global w1, b1, w2, b2

    z1 = np.dot(w1, x) + b1
    a1 = 1 / (1 + np.exp(-z1))

    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))

    return a1, a2


def backward(x, y, a1, a2):
    global w1, b1, w2, b2

    # da2 = (-y / a2) + (1 - y) / (1 - a2)
    dz2 = a2 - y
    dw2 = np.mean(np.dot(dz2, a1.T), axis=1, keepdims=True)
    db2 = np.mean(dz2, axis=1, keepdims=True)

    da1 = np.dot(w2.T, dz2)
    dz1 = da1 * a1 * (1 - a1)
    dw1 = np.mean(np.dot(dz1, x.T), axis=1, keepdims=True)
    db1 = np.mean(dz1, axis=1, keepdims=True)

    w1 -= task1.alpha * dw1
    b1 -= task1.alpha * db1
    w2 -= task1.alpha * dw2
    b2 -= task1.alpha * db2


def train(iteration, x, y):
    for i in range(iteration):
        a1, a2 = forward(x)
        backward(x, y, a1, a2)

    print('[Task 3] Train Accuracy: {}'.format(task1.accuracy(y, a2)))


def test(x, y):
    a1, a2 = forward(x)
    print('[Task 3] Test Accuracy: {}'.format(task1.accuracy(y, a2)))


if __name__ == '__main__':
    train(100, task1.x_train, task1.y_train)
    test(task1.x_test, task1.y_test)
