import numpy as np
from PIL import Image
from keras.callbacks import Callback
from deepsense import neptune

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
        # validation_predictions = self.model.predict_classes(self.x_test)
        # scores = self.model.predict(self.x_test)

        # Identify the incorrectly classified images and send them to Neptune Dashboard.
        # image_per_epoch = 0
        # for index, (prediction, actual) in enumerate(zip(validation_predictions, self.y_test)):
        #     if prediction != actual:
        #         if image_per_epoch == self.images_per_epoch:
        #             break
        #         image_per_epoch += 1
        #
        #         ctx.channel_send('false_predictions', neptune.Image(
        #             name='[{}] {} X {} V'.format(self.epoch_id, categories[prediction], categories[actual]),
        #             description="\n".join([
        #                 "{:5.1f}% {} {}".format(100 * score, categories[i], "!!!" if i == actual else "")
        #                 for i, score in enumerate(scores[index])]),
        #             data=array_2d_to_image(self.x_test[index,:,:])))
