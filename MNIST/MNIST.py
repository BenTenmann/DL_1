#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST: The "Hello World" of AI

@author: benjamintenmann
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Get the data as Numpy arrays
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Build a simple model
inputs = keras.Input(shape=(28, 28))
x = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
x = keras.layers.Flatten()(x)
#x = keras.layers.Dense(128, activation="relu")(x)
#x = keras.layers.Dense(128, activation="relu")(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()

# Compile the model
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy",
              metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

# Train the model for 1 epoch from Numpy data
batch_size = 128
print("Fit on NumPy data")
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=5)


# Train the model for 1 epoch using a dataset
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
print("Fit on Dataset")
history = model.fit(dataset, epochs=1)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
history = model.fit(dataset, epochs=1, validation_data=val_dataset)

loss, acc = model.evaluate(val_dataset)  # returns loss and metrics
print("loss: %.2f" % loss)
print("acc: %.2f" % acc)
