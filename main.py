from glob import glob
from matplotlib import pyplot as plt
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, Dense, MaxPool2D

from helpers import NeptuneCallback

from deepsense import neptune

ctx = neptune.Context()
# ctx.integrate_with_keras()

base_path = "../input/tryponet_set2.tar.gz/tryponet_set2/"

def load_Xy(path):
    filenames0 = glob(path + "norm/*.png")
    filenames1 = glob(path + "trypo/*.png")
    X = np.zeros((len(filenames0) + len(filenames1), 256, 256, 3))

    y = np.zeros(len(filenames0) + len(filenames1))
    y[len(filenames0):] = 1.

    print(path + "norm/*.png")
    for i, filename in enumerate(filenames0):
        X[i] = plt.imread(filename)
        if i % 100 == 0:
            print(i, end=" ")
        if i % 1000 == 0:
            print("")

    print("\n")
    print(path + "trypo/*.png")
    for i, filename in enumerate(filenames1):
        X[len(filenames0) + i] = plt.imread(filename)
        if i % 100 == 0:
            print(i, end=" ")
        if i % 1000 == 0:
            print("")

    return X, y

X_test, y_test = load_Xy(base_path + "valid/")
X_train, y_train = load_Xy(base_path + "train/")

model = Sequential()
model.add(Flatten(input_shape=(256, 256, 3)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
      epochs=10,
      batch_size=32,
      validation_data=(X_test, y_test),
      verbose=2,
      callbacks=[NeptuneCallback(X_test, y_test, images_per_epoch=20)])
