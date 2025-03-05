# Trained model 

# Model will be trained on TensorFlow


# Just a model to test how it interacts when hooking it up to a board 


import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, 3, padding="valid", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10)
    ]
)

def my_model():
  inputs = keras.Input(shape=(32, 32, 3))
  x = layers.Conv2D(
      32, 3, padding='same', kernel_regularizer=regularizers.l2(0.01)),(inputs)
  x = layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = layers.MaxPooling2D()
  x = layers.Conv2D(
      64, 5, padding ="same", kernel_regularizer=regularizers.l2(0.01)),(x)
  x = layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = layers.Conv2D(
      128, 3, padding='same', kernel_regularizer=regularizers.l2(0.01)),(x)
  x = layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = layers.Flatten()(x)
  x = layers.Dense(
      64, activation="relu", kernel_regularizer=regularizers.l2(0.01)),(x)
  x = layers.Dropout(0.5)(x)
  outputs = layers.Dense(10)(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model

  model = my_model()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=2)

model.evaluate(x_test, y_test, batch_size=64, verbose=2)

# Graphing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wave
import pylab
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix
import itertools

history_model1 = model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=2)

history_dict_model1 = history_model1.history
loss_values_model1 = history_dict_model1['loss']
accuracy_model1 = history_dict_model1['accuracy']
epochs_model1 = range(1, len(loss_values_model1) + 1)

plt.figure(figsize=(10, 8))
plt.plot(epochs_model1, accuracy_model1, label="Sequential API Accuracy", color="red")
plt.title("Model Performance Improvement per Epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

