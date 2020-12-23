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
from keras.layers import Activation, BatchNormalization, Input, Flatten, Reshape, Dense
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard
import pandas as pd
keras.backend.set_image_dim_ordering("th")
from PIL import Image
import glob
from sklearn.model_selection import train_test_split

DATASET_FILE = 'data/larger/'
OUT_DIR ='shallow_salnet_output/'

LEARNING_RATE = 1e-6


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
        rgb_images.append(np.array(Image.open(filename))[..., :3].T) # .png files have a pesky fourth transparency channel

    # # In the future will have to do some depth channel processing
    # for filename in glob.glob(DATASET_FILE + '*depth.png'):
    #     depth_images.append(np.array(Image.open(filename))[..., :3].T / 255.)

    # Need to turn suction images into 1 channel ground truth
    for filename in glob.glob(DATASET_FILE + '*suction.png'):
        label = np.where(np.array(Image.open(filename)).T > 0.5, 1, 0)
        label = label[np.newaxis, ...]
        suction_images.append(label)

    rgb_images = np.array(rgb_images)
    # depth_images = np.array(depth_images)
    suction_images = np.array(suction_images)

    # Normalize
    rgb_images = (rgb_images - np.mean(rgb_images)) / np.std(rgb_images)

    # Make the random split
    X_train, X_val, Y_train, Y_val, = train_test_split(rgb_images, suction_images, test_size=0.10, random_state=42)


    print('Training shapes X: {} and Y: {}'.format(X_train.shape, Y_train.shape))
    print('Validation shapes X: {} and Y: {}'.format(X_val.shape, Y_val.shape))
    return (X_train, Y_train), (X_val, Y_val)


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
    conv = Convolution2D(
        nb_filter, s, s,
        name=name,
        activation='linear',
        bias=False,
        border_mode='same',
        weights=[w])
    return lambda x: conv(upsample(x))

def get_model():

    input_tensor = Input(shape=(3, 224, 224)) 
    base_model = VGG16(weights='imagenet', input_tensor=input_tensor)

    x = base_model.get_layer('block3_conv3').output
    x = Convolution2D(256, 7, 7, init='normal', activation='relu', border_mode='same', name='conv6')(x)
    x = Convolution2D(128, 11, 11, init='normal', activation='relu', border_mode='same', name='conv7')(x)
    x = Convolution2D(32 , 11, 11, init='normal', activation='relu', border_mode='same', name='conv8')(x)
    x = Convolution2D(1 , 13, 13, init='normal', activation='relu', border_mode='same', name='conv9')(x)
    x = deconv(1,4, 'deconv')(x)
    output = Activation('sigmoid')(x)


    model = Model(input=input_tensor, output=output)

    for layer in base_model.layers:
        layer.trainable = False

    sgd = SGD(lr=LEARNING_RATE, decay=0.005, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mean_squared_error')
    return model

def save_outputs(outputs):
    for idx, output in enumerate(outputs):
        output = output[0, ...]
        print('Saving output image {}'.format(idx))
        img = Image.fromarray(output, 'L')
        img.save(OUT_DIR + str(idx) + '.png')


def main():
    model = get_model()
    model.summary()

    # model.load_weights(OUT_DIR + 'model.01.weights')

    (X_train, Y_train), val = load_datasets()

    #### Predict ####
    print('Starting prediction')
    predictions = model.predict(val[0], verbose=1)

    print('Prediction shape is {}'.format(predictions.shape))
    print(predictions)

    save_outputs(predictions)

    ################

    # checkpointer = ModelCheckpoint(
    #     OUT_DIR + 'model.weights',
    #     monitor='val_loss',
    #     save_best_only=False,
    #     period=10)

    # lr_scheduler = LearningRateScheduler(
    #     schedule=PeriodicLR(LEARNING_RATE, 100, 0.5))

    # tensorboard = TensorBoard(histogram_freq=1, write_grads=True)

    # try:
    #     model.fit(
    #         X_train,
    #         Y_train,
    #         batch_size=2,
    #         epochs=100_000,
    #         shuffle="batch",
    #         validation_data = val,
    #         callbacks=[checkpointer, lr_scheduler, tensorboard])

    # finally:

    #     if hasattr(model, 'history'):
    #         history = pd.DataFrame(model.history.history)
    #         history.to_csv(OUT_DIR + 'train_salvol20.log', index_label='epoch')
    #         print(history)

    #     try:
    #         model.save_weights(OUT_DIR + 'final.weights')
    #         print('Saved final weights in snapshots/salvol20/final.weights')
    #     except Exception as e:
    #         print(e)


if __name__ == '__main__':
    main()