#!/usr/bin/env python3

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import tensorflow as tf
from tensorflow import keras

test_range = (-40, 40)
graph_range = (-40, 40)

def generate_random_data(n = 1000):
  x = np.random.rand(n, 1)
  x *= test_range[1] - test_range[0]
  x += test_range[0]
  y = np.sin(x)
  return x, y

model = keras.Sequential([
  keras.layers.Dense(50, activation='elu'),
  keras.layers.Dense(50, activation='elu'),
  keras.layers.Dense(50, activation='elu'),
  keras.layers.Dense(50, activation='elu'),
  keras.layers.Dense(50, activation='elu'),
  keras.layers.Dense(50, activation='elu'),
  keras.layers.Dense(50, activation='elu'),
  keras.layers.Dense(50, activation='elu'),
  keras.layers.Dense(50, activation='elu'),
  keras.layers.Dense(1, activation=None),
])

model.compile(
  optimizer='adam',
  loss='mean_squared_error'
)

def fit():
  x, y = generate_random_data()
  model.fit(x, y, batch_size=200, epochs=1, verbose=0)

for _ in range(100):
  fit()

graph_x = np.arange(graph_range[0], graph_range[1], 0.1)
graph_y = np.sin(graph_x)

fig, ax = plt.subplots()
ln0, = plt.plot(graph_x, graph_y)
ln1, = plt.plot([], [])

def init():
    ax.set_xlim(graph_range[0], graph_range[1])
    ax.set_ylim(-1, 1)
    return ln0, ln1
def update(frame):
    fit()
    pred_y = model.predict(graph_x.reshape(-1, 1)).flatten()
    diff = np.mean(np.abs(graph_y - pred_y))
    print('diff:', diff)
    ln1.set_data(graph_x, pred_y)
    return ln0, ln1

ani = FuncAnimation(fig, update, init_func=init, interval=1, blit=True)
plt.show()



