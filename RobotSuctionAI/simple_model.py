'''
   Implementation of shallow SalNet
'''
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Reshape, Flatten

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard

# import numpy as np
# import keras
# from keras.optimizers import SGD
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.applications.vgg16 import VGG16
# from keras.layers import Input, Activation
# from keras.layers import Activation, BatchNormalization, Input
# from keras.layers.convolutional import Convolution2D, UpSampling2D
# from keras.models import Model

import pandas as pd

keras.backend.set_image_dim_ordering("tf")
from PIL import Image
import glob
from sklearn.model_selection import train_test_split

DATASET_FILE = 'data/larger/'
OUT_DIR ='output/salnet/large_simple3/'

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

    for filename in glob.glob(DATASET_FILE + '*rgb.png'):
        rgb_images.append(np.array(Image.open(filename))[..., :3]) # .png files have a pesky fourth transparency channel

    # In the future will have to do some depth channel processing
    # for filename in glob.glob(DATASET_FILE + '*depth.png'):
    #     depth_images.append(np.array(Image.open(filename))[..., :3] / 255.)

    # Need to turn suction images into 1 channel ground truth
    for filename in glob.glob(DATASET_FILE + '*suction.png'):
        im = Image.open(filename).resize((48, 48))
        label = np.where(np.array(im) > 0.5, 1, 0)
        label = label[...]
        suction_images.append(label)

    rgb_images = np.array(rgb_images)
    suction_images = np.array(suction_images)

    # Normalize
    rgb_images = (rgb_images - np.mean(rgb_images)) / np.std(rgb_images)

    # Make the random split
    X_train, X_val, Y_train, Y_val, = train_test_split(rgb_images, suction_images, test_size=0.20, random_state=42)


    print('Training shapes X: {} and Y: {}'.format(X_train.shape, Y_train.shape))
    print('Validation shapes X: {} and Y: {}'.format(X_val.shape, Y_val.shape))
    return (X_train, Y_train), (X_val, Y_val)

def get_model():
    model = Sequential()

    model.add(Conv2D(32, 5, padding='same', activation='relu', input_shape=(96, 96, 3)))
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=3, strides=2))
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=3, strides=2))
    model.add(Flatten())
    model.add(Dense(4608))
    model.add(Dense(2304))
    model.add(Reshape((48, 48)))


    sgd = SGD(decay=0.005, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=sgd,
        loss='mean_squared_error'
        )

    return model

def save_outputs(outputs):
    for idx, output in enumerate(outputs):
        print('Saving output image {}'.format(idx))
        img = Image.fromarray(output, 'L')
        img.save(OUT_DIR + str(idx) + '.png')


def main():
    model = get_model()
    model.summary()

    # model.load_weights(OUT_DIR + 'final.weights')

    (X_train, Y_train), val = load_datasets()

    checkpointer = ModelCheckpoint(
        OUT_DIR + 'model.weights',
        monitor='val_loss',
        save_best_only=True)

    lr = 0.001
    lr_scheduler = LearningRateScheduler(
        schedule=PeriodicLR(lr, 10_000, 0.5))

    tensorboard = TensorBoard(histogram_freq=1, write_grads=True)

    try:
        model.fit(
            X_train,
            Y_train,
            batch_size=4,
            nb_epoch=100,
            shuffle="batch",
            validation_data = val,
            callbacks=[checkpointer, lr_scheduler, tensorboard])

    finally:

        if hasattr(model, 'history'):
            history = pd.DataFrame(model.history.history)
            history.to_csv(OUT_DIR + 'train_.log', index_label='epoch')
            print(history)

        try:
            model.save_weights(OUT_DIR + 'final.weights')
            print('Saved final weights')
        except Exception as e:
            print(e)

    #### Predict ####
    # print('Starting prediction')
    # predictions = model.predict(val[0], verbose=1)

    # print('Prediction shape is {}'.format(predictions.shape))
    # print(predictions)

    # save_outputs(predictions)

    ################

if __name__ == '__main__':
    main()