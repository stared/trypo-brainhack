import numpy as np
from PIL import Image
from keras.callbacks import Callback
from deepsense import neptune
from glob import glob
from matplotlib import pyplot as plt

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

ctx = neptune.Context()

def array_2d_to_image(array, autorescale=True):
    assert array.min() >= 0
    assert len(array.shape) in [2, 3]
    if array.max() <= 1 and autorescale:
        array = 255 * array
    array = array.astype('uint8')
    return Image.fromarray(array)

def model_summary(model):
    print("Model created successfully.")
    print(model.summary())
    ctx.channel_send('n_layers', len(model.layers))
    ctx.channel_send('n_parameters', model.count_params())

categories = ['norm', 'trypo']

class NeptuneCallback(Callback):
    def __init__(self, x_test, y_test, images_per_epoch=-1):

        try:
            ctx.channel_reset('Log-loss training')
            ctx.channel_reset('Log-loss validation')
            ctx.channel_reset('Accuracy training')
            ctx.channel_reset('Accuracy validation')
            ctx.channel_reset('false_predictions')
        except:
            pass
        self.epoch_id = 0
        self.images_per_epoch = images_per_epoch
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1

        # logging numeric channels
        ctx.channel_send('Log-loss training', self.epoch_id, logs['loss'])
        ctx.channel_send('Log-loss validation', self.epoch_id, logs['val_loss'])
        ctx.channel_send('Accuracy training', self.epoch_id, logs['acc'])
        ctx.channel_send('Accuracy validation', self.epoch_id, logs['val_acc'])

        # Predict the digits for images of the test set.
        validation_predictions = self.model.predict_classes(self.x_test)
        scores = self.model.predict(self.x_test)

        # Identify the incorrectly classified images and send them to Neptune Dashboard.
        image_per_epoch = 0
        for index, (prediction, actual) in enumerate(zip(validation_predictions, self.y_test.argmax(axis=1))):
            if prediction != actual:
                if image_per_epoch == self.images_per_epoch:
                    break
                image_per_epoch += 1

                ctx.channel_send('false_predictions', neptune.Image(
                    name='[{}] {} X {} V'.format(self.epoch_id, categories[prediction], categories[actual]),
                    description="\n".join([
                        "{:5.1f}% {} {}".format(100 * score, categories[i], "!!!" if i == actual else "")
                        for i, score in enumerate(scores[index])]),
                    data=array_2d_to_image(self.x_test[index,:,:])))
