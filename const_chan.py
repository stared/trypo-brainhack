import keras
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, Dense, MaxPool2D, GlobalAveragePooling2D, Dropout
from keras import utils

import numpy as np
from PIL import Image
from helpers import NeptuneCallback, model_summary, load_Xy
from deepsense import neptune
ctx = neptune.Context()

base_path = "../input/tryponet_set2.tar.gz/tryponet_set2/"

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPool2D())

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(64, (1, 1), activation='relu'))
model.add(GlobalAveragePooling2D())

model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_summary(model)

X_test, y_test = load_Xy(base_path + "valid/")
X_test = np.append(X_test[::2], X_test[1::2], axis=0)
y_test = np.append(y_test[::2], y_test[1::2], axis=0)

X_train, y_train = load_Xy(base_path + "train/")
Y_train = utils.to_categorical(y_train, 2)
Y_test = utils.to_categorical(y_test, 2)

model.fit(X_train, Y_train,
      epochs=50,
      batch_size=32,
      validation_data=(X_test, Y_test),
      verbose=2,
      callbacks=[NeptuneCallback(X_test, Y_test, images_per_epoch=20)])
