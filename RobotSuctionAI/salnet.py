'''
    Model: model1
'''
from keras.optimizers import SGD
import numpy as np
import keras
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Activation
from keras.layers import Activation, BatchNormalization, Input
from keras.layers.convolutional import Convolution2D, Conv2D, UpSampling2D
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.metrics import mean_iou
import tensorflow as tf
from keras import regularizers
import pandas as pd
import keras.backend as K
keras.backend.set_image_dim_ordering("th")
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
import sys

DATA_DIRECTORY = 'data/larger/'
OUT_DIR ='large_aws_output/'

LEARNING_RATE = 5e-8

def keras_mean_iou(y_true, y_pred):
    return tf.metrics.mean_iou(y_true, y_pred, 2)[1]


class PeriodicLR(object):
    """
    Learning rate schedule that periodically reduces
    """
    def __init__(self, base_lr, epochs, gamma):
        self.base_lr = base_lr
        self.epochs = epochs
        self.gamma = gamma

    def __call__(self, epoch):
        n = epoch / self.epochs
        return self.base_lr * (self.gamma ** n)


def load_datasets():
    rgb_images = []
    depth_images = []
    suction_images = []

    for rgb_filename in glob.glob(DATA_DIRECTORY + '*-rgb.png'):

        file_id = rgb_filename[len(DATA_DIRECTORY): -8]

        suction_filename = DATA_DIRECTORY + file_id + '-suction.png'

        rgb_images.append(np.array(Image.open(rgb_filename))[..., :3].T) # .png files have a pesky fourth transparency channel
        label = np.array(Image.open(suction_filename)).T
        label = label[np.newaxis, ...]
        suction_images.append(label)

    rgb_images = np.array(rgb_images)

    suction_images = np.array(suction_images)

    return rgb_images, suction_images

def norm_weights(n):
    r = n / 2.0
    xs = np.linspace(-r, r, n)
    x, y = np.meshgrid(xs, xs)
    w = np.exp(-0.5*(x**2 + y**2))
    w /= w.sum()
    return w

def deconv(nb_filter, size, name):
    upsample = UpSampling2D(size=(size, size))
    s = 2 * size + 1
    w = norm_weights(s)[:, :, np.newaxis, np.newaxis]
    conv = Conv2D(
        nb_filter, (s, s),
        name=name,
        activation='linear',
        bias=False,
        border_mode='same',
        weights=[w],
        kernel_regularizer=regularizers.l2(0.01))
    return lambda x: conv(upsample(x))

def get_model():

    input_tensor = Input(shape=(3, 224, 224)) 
    base_model = VGG16(weights='imagenet', input_tensor=input_tensor)

    x = base_model.get_layer('block3_conv3').output
    x = Conv2D(512, (5, 5), activation="relu", name="conv4", padding="same", kernel_regularizer=regularizers.l2(0.01))(x)
    x = Conv2D(512, (5, 5), activation="relu", name="conv5", padding="same", kernel_regularizer=regularizers.l2(0.01))(x)
    x = Conv2D(256, (7, 7), activation="relu", name="conv6", padding="same", kernel_regularizer=regularizers.l2(0.01))(x)
    x = Conv2D(128, (11, 11), activation="relu", name="conv7", padding="same", kernel_regularizer=regularizers.l2(0.01))(x)
    x = Conv2D(32, (11, 11), activation="relu", name="conv8", padding="same", kernel_regularizer=regularizers.l2(0.01))(x)
    x = Conv2D(1, (13, 13), activation="relu", name="conv9", padding="same", kernel_regularizer=regularizers.l2(0.01))(x)
    x = deconv(1,4, 'deconv')(x)
    output = Activation('sigmoid')(x)

    model = Model(input=input_tensor, output=output)

    for layer in base_model.layers:
        layer.trainable = False

    sgd = SGD(lr=LEARNING_RATE, decay=0.005, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='mean_squared_error', metrics = [keras_mean_iou])
    return model

def save_predictions(inputs, outputs, predictions):
    for idx, prediction in enumerate(predictions):
        print('Saving prediction image {}'.format(idx))

        prediction = prediction[0, ...].T
        img = Image.fromarray((prediction * 255).astype(np.uint8), 'L')
        img.save(OUT_DIR + str(idx) + 'prediction.png')

        img = Image.fromarray((outputs[idx][0, ...].T * 255).astype(np.uint8), 'L')
        img.save(OUT_DIR + str(idx) + 'label.png')

        img = Image.fromarray(inputs[idx].T, 'RGB')
        img.save(OUT_DIR + str(idx) + '.png')

def preprocess(inputs, outputs):
    X = inputs / 255.

    Y = outputs / 149.

    return X, Y

def main():
    model = get_model()
    model.summary()

    inputs, outputs = load_datasets()

    X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = train_test_split(inputs, outputs, test_size=0.10, random_state=16)

    X_train, Y_train = preprocess(X_train_orig, Y_train_orig)
    X_test, Y_test = preprocess(X_test_orig, Y_test_orig)

    if len(sys.argv) != 1 and sys.argv[1] == 'predict':
        model.load_weights(OUT_DIR + 'final.weights')
        print('Starting prediction')
        predictions = model.predict(X_test, verbose=1)

        print('Prediction shape is {}'.format(predictions.shape))

        save_predictions(X_test_orig, Y_test, predictions)

    else:
        K.get_session().run(tf.local_variables_initializer())

        class GeneratePredictions(Callback):
            def on_epoch_end(self, batch, logs={}):
                pass

        checkpointer = ModelCheckpoint(
            OUT_DIR + 'model.weights',
            monitor='val_loss',
            save_best_only=True,
            period=10)


        lr_scheduler = LearningRateScheduler(
            schedule=PeriodicLR(LEARNING_RATE, 2, 0.5))

        tensorboard = TensorBoard(histogram_freq=1, write_grads=True)

        predictioner = GeneratePredictions()

        try:
            model.fit(
                X_train,
                Y_train,
                batch_size=2,
                epochs=100_000,
                shuffle="batch",
                validation_data = (X_test, Y_test),
                callbacks=[checkpointer, lr_scheduler, tensorboard, predictioner])

        finally:

            if hasattr(model, 'history'):
                history = pd.DataFrame(model.history.history)
                history.to_csv(OUT_DIR + 'train.log', index_label='epoch')
                print(history)

            try:
                model.save_weights(OUT_DIR + 'final.weights')
                print('Saved final weights in snapshots/salvol20/final.weights')
            except Exception as e:
                print(e)


if __name__ == '__main__':
    main()