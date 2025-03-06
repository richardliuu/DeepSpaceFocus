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