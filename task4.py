import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

# Practice 4
# m = 1000, n = 100, iteration = 1000
# Activation Function : Sigmoid 

def generate(m, dim, rng):
    x = np.random.uniform(rng[0], rng[1], size=(m, dim))  # x : (m, dim)
    y = []
    for i in range(m):
        if x[i][0] ** 2 > x[i][1]:
            y.append(1)
        else:
            y.append(0)
    y = np.array(y).reshape((m, 1))  # y : (m, 1)
    return x, y

x_train, y_train = generate(1000, 2, [-2, 2])
x_test, y_test = generate(100, 2, [-2, 2])

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(2,)),
    tf.keras.layers.Dense(3, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='binary_crossentropy', metrics=['accuracy'])

# Table 1
# model.compile(optimizer=tf.keras.optimizers.SGD(0.1), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.SGD(0.1), loss='mean_squared_error', metrics=['accuracy'])

# Table 2
# model.compile(optimizer=tf.keras.optimizers.SGD(0.1), loss='mean_squared_error', metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.RMSprop(0.1), loss='mean_squared_error', metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error', metrics=['accuracy'])

# Table 3
# model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='binary_crossentropy', metrics=['accuracy'])


train_start = time.time()
train_result = model.fit(x_train, y_train, batch_size=1000, epochs=1000, verbose=0)
train_end = time.time() - train_start
print('Train accuracy: %f' % train_result.history['accuracy'][-1])
print('Train loss: %f' % train_result.history['loss'][-1])
print('Train time: %f' % train_end)

print('\n')

test_start = time.time()
test_result = model.evaluate(x_test, y_test, batch_size=1000, verbose=0) # batch_size = 100
test_end = time.time() - test_start
print('Test accuracy: %f' % test_result[1])
print('Test loss: %f' % test_result[0])
print('Test time: %f' % test_end)