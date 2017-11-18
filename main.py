import keras
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, Dense, MaxPool2D, GlobalAveragePooling2D
from keras import utils

from glob import glob
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from helpers import NeptuneCallback, model_summary
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

model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPool2D())

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(64, (1, 1), activation='relu'))
model.add(GlobalAveragePooling2D())

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_summary(model)

X_test, y_test = load_Xy(base_path + "valid/")
X_train, y_train = load_Xy(base_path + "train/")
Y_train = utils.to_categorical(y_train, 2)
Y_test = utils.to_categorical(y_test, 2)

model.fit(X_train, Y_train,
      epochs=50,
      batch_size=32,
      validation_data=(X_test, Y_test),
      verbose=2,
      callbacks=[NeptuneCallback(X_test, Y_test, images_per_epoch=20)])
