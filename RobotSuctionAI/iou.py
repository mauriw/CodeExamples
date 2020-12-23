import numpy as np
import glob
from PIL import Image

DATA_DIRECTORY = "small_fcnn/"

EPSILON = 1e-12

def load_datasets():
    prediction_images = []
    label_images = []

    image_type = '_prediction.png' # '-rgb.png'
    for prediction in glob.glob(DATA_DIRECTORY + '*' + image_type):
        file_id = prediction[len(DATA_DIRECTORY): -1 * len(image_type)]

        label_filename = DATA_DIRECTORY + file_id + '_label.png'

        prediction_images.append(np.array(Image.open(prediction))) # .png files have a pesky fourth transparency channel
        label_images.append(np.array(Image.open(label_filename))) # .png files have a pesky fourth transparency channel

    prediction_images = np.array(prediction_images, dtype = float)

    label_images = np.array(label_images)

    return prediction_images, label_images

def preprocess(inputs, outputs):
    X = inputs / 255.

    Y = outputs / 149.

    return X, Y

def mean_iou(predictions, y_label):
    num_examples = predictions.shape[0]
    y_pred = (predictions > 0.5)
    y_pred = y_pred.reshape((num_examples, -1))
    y_label = y_label.reshape((num_examples, -1))

    true_positives = np.count_nonzero(np.logical_and(y_pred == 1, y_label ==1), axis = 1)
    positives = np.count_nonzero(y_label, axis = 1)
    pred_positives = np.count_nonzero(y_pred, axis = 1)
    union = positives + pred_positives - true_positives
    iou = np.divide(true_positives, union + EPSILON)
    return np.mean(iou)

def main():
    prediction_images, label_images = load_datasets()
    print(prediction_images.shape)
    print(label_images.shape)
    prediction_images = prediction_images/255
    label_images = label_images / 255
    iou = mean_iou(prediction_images, label_images)
    print("Mean_iou: ", iou)

if __name__ == '__main__':
    main()
