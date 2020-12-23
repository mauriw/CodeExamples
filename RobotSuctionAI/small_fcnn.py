
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import glob
import tensorflow as tf
import scipy
import os

DATA_DIRECTORY = 'data/augmented/'
OUT_DIR ='rgb_fcnn/'
EXPORT_DIR = 'small_fcnn_model/'
IMAGE_SHAPE = (224, 224)

EPOCHS = 80
BATCH_SIZE = 4
VGG_PATH = 'data/vgg'
KEEP_PROB = 0.5
LEARNING_RATE = 1e-4
OUTPUT_DIM = 1
HALF_LEARNING_RATE_EVERY = 0
EPOCHS_BETWEEN_CHECKPOINTS = 10

def load_vgg(sess):

    # load the model and weights
    model = tf.saved_model.loader.load(sess, ['vgg16'], VGG_PATH)

    # Get Tensors to be returned from graph

    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    layer3 = graph.get_tensor_by_name('layer3_out:0')
    layer4 = graph.get_tensor_by_name('layer4_out:0')
    layer7 = graph.get_tensor_by_name('layer7_out:0')

    return image_input, keep_prob, layer3, layer4, layer7


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out):

    # Use a shorter variable name for simplicity
    layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

    # Apply 1x1 convolution in place of fully connected layer
    fcn8 = tf.layers.conv2d(layer7, filters=OUTPUT_DIM, kernel_size=1, name="fcn8")

    # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so that we can add skip connection with 4th layer
    fcn9 = tf.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1],
    kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")

    # Add a skip connection between current final layer fcn8 and 4th layer
    fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")

    # Upsample again
    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
    kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")

    # Add skip connection
    fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

    # Upsample again
    fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=OUTPUT_DIM,
    kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")

    return fcn11


def optimize(nn_last_layer, correct_label, learning_rate):

    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(nn_last_layer, (-1, OUTPUT_DIM), name="fcn_logits")
    correct_label_reshaped = tf.reshape(correct_label, (-1, OUTPUT_DIM))

    # Calculate distance from actual labels using cross entropy
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped)
    # Take mean for total loss
    loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

    # The model implements this operation to find the weights/parameters that would yield correct pixel labels
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

    return logits, train_op, loss_op

def train(sess, train_op, cross_entropy_loss,
            input_image, logits, correct_label,
            keep_prob, learning_rate,
            X_train, Y_train, X_val, Y_val, writer, X_val_orig):

    saver = tf.train.Saver(max_to_keep=1)

    training_loss_placeholder = tf.placeholder(tf.float32)
    validation_loss_placeholder = tf.placeholder(tf.float32)
    lr_placeholder = tf.placeholder(tf.float32)

    iter_train_loss_placeholder = tf.placeholder(tf.float32)
    iter_train_loss_op = tf.summary.scalar('iter_training_loss', iter_train_loss_placeholder)

    epoch_tb_summ = tf.summary.merge([tf.summary.scalar('training_loss', training_loss_placeholder),
                                     tf.summary.scalar('validation_loss', validation_loss_placeholder),
                                     tf.summary.scalar('learning_rate', lr_placeholder)])

    # TODO: Mean iou struggles to work over batches.    JOSE FIX THIS
    # indiv_Y_val_placeholder = tf.placeholder(tf.float32)
    # indiv_predictions_placeholder = tf.placeholder(tf.float32)
    # iou_tensor, _ = tf.metrics.mean_iou(indiv_Y_val_placeholder, indiv_predictions_placeholder, 2)
    # iou_summ = tf.summary.scalar('val_mean_iou', iou_tensor)

    iterations = 0
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):

        print("EPOCH {}: ".format(epoch + 1))

        # Shuffling the data
        train_shuffled_indices = np.arange(X_train.shape[0])
        val_shuffled_indices = np.arange(X_val.shape[0])
        np.random.shuffle(train_shuffled_indices)
        np.random.shuffle(val_shuffled_indices)

        # Getting the new learning rate
        if HALF_LEARNING_RATE_EVERY != 0:
            learning_rate_value = LEARNING_RATE * (0.5 ** epoch / HALF_LEARNING_RATE_EVERY)
        else:
            learning_rate_value = LEARNING_RATE

        # Training step
        training_loss = 0.0

        for i in range(0, X_train.shape[0], BATCH_SIZE):

            X_batch = X_train[i:i + BATCH_SIZE]
            Y_batch = Y_train[i:i + BATCH_SIZE]

            loss, _ = sess.run([cross_entropy_loss, train_op],
                                  feed_dict={input_image: X_batch, correct_label: Y_batch,
                                             keep_prob: KEEP_PROB, learning_rate:learning_rate_value})

            training_loss += loss

            # Writing the per iteration tensorboard things
            iter_summary = sess.run(iter_train_loss_op, feed_dict={iter_train_loss_placeholder: loss / X_batch.shape[0]})
            writer.add_summary(iter_summary, iterations)

            iterations += 1

        # Evaluation step
        predictions, val_loss = predict(sess, input_image, logits, keep_prob, X_val, Y_val, cross_entropy_loss, correct_label)
        iou = mean_iou(predictions, Y_val)

        # Compute mean iou
        # iou = sess.run(iou_summ, feed_dict={Y_val_placeholder: Y_val, predictions_placeholder: predictions})
        # writer.add_summary(iou, epoch)
        # TODO: JOSE FIX THIS

        training_loss /= X_train.shape[0]
        val_loss /= X_val.shape[0]

        print("Training averaged loss = {}".format(training_loss))
        print("Validation averaged loss = {}".format(val_loss))
        print("Validation mean_iou :", iou)

        # Writing the per epoch tensorboard things
        s = sess.run(epoch_tb_summ, feed_dict={training_loss_placeholder: training_loss, validation_loss_placeholder: val_loss, lr_placeholder: learning_rate_value})
        writer.add_summary(s, epoch)

        # Checkpoint by saving val predictions and the model
        if epoch % EPOCHS_BETWEEN_CHECKPOINTS == 0:
            path = OUT_DIR + 'epoch_' + str(epoch) + "/"

            if not os.path.exists(path):
                os.makedirs(path)

            save_predictions(X_val_orig, Y_val, predictions, path)

            # Save the model
            if val_loss <= best_val_loss:
                best_val_loss = val_loss

                saver.save(sess, OUT_DIR + 'model_checkpoint', global_step=epoch)

                print("Saved the model_checkpoint")

def mean_iou(predictions, y_label):
    num_examples = predictions.shape[0]
    y_pred = (predictions > 0.5)
    y_pred = y_pred.reshape((num_examples, -1))
    y_label = y_label.reshape((num_examples, -1))

    true_positives = np.count_nonzero(np.logical_and(y_pred == 1, y_label ==1), axis = 1)
    positives = np.count_nonzero(y_label, axis = 1)
    pred_positives = np.count_nonzero(y_pred, axis = 1)
    union = positives + pred_positives - true_positives
    iou = np.divide(true_positives, union)
    return np.mean(iou)


def predict(sess, input_image, logits, keep_prob, X, Y, cross_entropy_loss, correct_label):

    predictions, loss = sess.run([logits, cross_entropy_loss], feed_dict={input_image: X, keep_prob: 1.0, correct_label: Y})

    predictions = np.reshape(predictions, (X.shape[0], IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1))

    predictions = scipy.special.expit(predictions)

    return predictions, loss

def save_predictions(inputs, outputs, predictions, path=OUT_DIR):
    for idx, prediction in enumerate(predictions):
        prediction = prediction[..., 0]
        img = Image.fromarray((prediction * 255).astype(np.uint8), 'L')
        img.save(path + str(idx) + '_prediction.png')

        img = Image.fromarray((outputs[idx][..., 0] * 255).astype(np.uint8), 'L')
        img.save(path + str(idx) + '_label.png')

        img = Image.fromarray(inputs[idx], 'RGB')
        img.save(path + str(idx) + '_input.png')

    print('Saved {} prediction images'.format(len(predictions)))

def load_datasets():
    rgb_images = []
    suction_images = []

    image_type = '-rgb.png'
    for rgb_filename in glob.glob(DATA_DIRECTORY + '*' + image_type):

        file_id = rgb_filename[len(DATA_DIRECTORY): -1 * len(image_type)]

        suction_filename = DATA_DIRECTORY + file_id + '-suction.png'

        rgb_images.append(np.array(Image.open(rgb_filename))[..., :3]) # .png files have a pesky fourth transparency channel
        label = np.array(Image.open(suction_filename))
        label = label[..., np.newaxis]
        suction_images.append(label)

    rgb_images = np.array(rgb_images)

    suction_images = np.array(suction_images)

    return rgb_images, suction_images

def preprocess(inputs, outputs):
    X = inputs / 255.

    Y = outputs / 149.

    return X, Y

def run():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    inputs, outputs = load_datasets()

    X_train_orig, X_val_orig, Y_train_orig, Y_val_orig = train_test_split(inputs, outputs, test_size=0.10, random_state=3)

    X_train, Y_train = preprocess(X_train_orig, Y_train_orig)
    X_val, Y_val = preprocess(X_val_orig, Y_val_orig)

    with tf.Session() as session:

        # Tensorboard
        tf_writer = tf.summary.FileWriter(OUT_DIR, session.graph)

        # Returns the three layers, keep probability and input layer from the vgg architecture
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session)

        correct_label = tf.placeholder(tf.float32, shape=(None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], OUTPUT_DIM))
        learning_rate = tf.placeholder(tf.float32)

        # The resulting network architecture from adding a decoder on top of the given vgg model
        model_output = layers(layer3, layer4, layer7)

        # Returns the output logits, training operation and cost operation to be used
        # - logits: each row represents a pixel, each column a class
        # - train_op: function used to get the right parameters to the model to correctly label the pixels
        # - cross_entropy_loss: function outputting the cost which we are minimizing, lower cost should yield higher accuracy
        logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate)

        # Initialize all variables
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        print("Model build successful, starting training")

        try:
            # Train the neural network
            train(session, train_op, cross_entropy_loss, image_input, logits, correct_label, keep_prob, learning_rate, X_train, Y_train, X_val, Y_val, tf_writer, X_val_orig)

        finally:
            
            predictions, loss = predict(session, image_input, logits, keep_prob, X_val, Y_val, cross_entropy_loss, correct_label)

            print("Final validation loss was {}".format(loss))
            save_predictions(X_val_orig, Y_val, predictions)

            tf.saved_model.simple_save(session, EXPORT_DIR, inputs={'image_input:0': image_input}, outputs={"fcn_logits": logits})

if __name__ == '__main__':
    run()

"""
TODO:

TO JOSE: Search for JOSE FIX THIS in the code :P (mainly mean iou is an issue and I couldn't figure out what hacky thing to do)

Namespaces? For when we need to combine them
A DEPTH MODEL

add the IOU metric
Do we need a decreasing learning rate or not? (Seems like we don't)

DONE:
Checkpoints
Saving Models
Fix logging

"""