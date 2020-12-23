import numpy as np
import glob
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

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

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax  

def my_confusion_matrix(predictions, y_label):
    num_examples = predictions.shape[0]
    y_pred = (predictions > 0.5).astype(int)
    y_pred = y_pred.flatten()
    y_label = y_label.flatten().astype(int)

    confusionMatrix = confusion_matrix(y_label, y_pred)
    plot_confusion_matrix(y_label, y_pred, classes = ['No Suction', 'Suction'], normalize = True)
    print("Printing pixel ratios in labels")
    numFalse = confusionMatrix[0][0] + confusionMatrix[0][1]
    numTrue = confusionMatrix[1][0] + confusionMatrix[1][1]
    print("Suction pixels :", numTrue)
    print("No Suction pixels :", numFalse)
    print("Ratio suction/no suction: ", numTrue/numFalse)
    precission, recall, f_score, support = precision_recall_fscore_support(y_label, y_pred, average='macro')
    print("Precission: ", precission)
    print("Recall: ", recall)
    print("F score: ", f_score)
    print("Support ", support)

def main():
    prediction_images, label_images = load_datasets()
    print(prediction_images.shape)
    print(label_images.shape)
    prediction_images = prediction_images/255.
    label_images = label_images / 255
    iou = mean_iou(prediction_images, label_images)
    print("Mean_iou: ", iou)
    my_confusion_matrix(prediction_images, label_images)
    plt.show()

if __name__ == '__main__':
    main()
