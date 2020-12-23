import small_fcnn
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import glob
import os
from error_analysis import mean_iou
import scipy

OUT_DIR = 'rgbd_fcnn/'
DATA_DIRECTORY = 'data/augmented/'
EXPORT_DIR = 'rgbd_fcnn_model/'
IMAGE_SHAPE = (224, 224)

EPOCHS = 80
BATCH_SIZE = 4
VGG_PATH = 'data/vgg'
KEEP_PROB = 0.5
LEARNING_RATE = 1e-5
OUTPUT_DIM = 1
HALF_LEARNING_RATE_EVERY = 0
EPOCHS_BETWEEN_CHECKPOINTS = 10

def load_datasets():
	rgb_images = []
	depth_images = []
	label_images = []
	print("Loading Dataset")
	image_type = '-rgb.png' 
	for rgb_filename in glob.glob(DATA_DIRECTORY + '*' + image_type):

		file_id = rgb_filename[len(DATA_DIRECTORY): -1 * len(image_type)]

		label_filename = DATA_DIRECTORY + file_id + '-suction.png'
		depth_filename = DATA_DIRECTORY + file_id + '-depth.png'

		rgb_images.append(np.array(Image.open(rgb_filename))[..., :3]) # .png files have a pesky fourth transparency channel
		depth_images.append(np.array(Image.open(depth_filename)))
		label = np.array(Image.open(label_filename))
		label = label[..., np.newaxis]
		label_images.append(label)

	rgb_images = np.array(rgb_images)
	depth_images = np.array(depth_images)
	label_images = np.array(label_images)
	return rgb_images, depth_images, label_images

def preprocess(rgb, depth, outputs):
	X_rgb = rgb / 255.
	X_depth = depth / 255.

	Y = outputs / 149.

	return X_rgb, X_depth, Y


def train(sess, train_op, cross_entropy_loss,
			rgb_input, depth_input, logits, correct_label,
			rgb_keep_prob, depth_keep_prob, learning_rate,
			X_rgb_train, X_depth_train, Y_train, X_rgb_val, X_depth_val, Y_val, writer, X_rgb_val_orig, X_depth_val_orig):
	
	# saver = tf.train.Saver(max_to_keep=1)

	saver = tf.train.Saver(max_to_keep=1)

	training_loss_placeholder = tf.placeholder(tf.float32)
	validation_loss_placeholder = tf.placeholder(tf.float32)
	lr_placeholder = tf.placeholder(tf.float32)
	val_mean_iou_placeholder = tf.placeholder(tf.float32)

	epoch_tb_summ = tf.summary.merge([tf.summary.scalar('training_loss', training_loss_placeholder),
									 tf.summary.scalar('validation_loss', validation_loss_placeholder),
									 tf.summary.scalar('learning_rate', lr_placeholder),
									 tf.summary.scalar('val_mean_iou', val_mean_iou_placeholder)])

	iterations = 0
	best_val_loss = float('inf')

	for epoch in range(EPOCHS):

		print("EPOCH {}: ".format(epoch + 1))

		# Shuffling the data
		# TO DO: how do we use these?
		train_shuffled_indices = np.arange(X_rgb_train.shape[0])
		val_shuffled_indices = np.arange(X_rgb_val.shape[0])
		np.random.shuffle(train_shuffled_indices)
		np.random.shuffle(val_shuffled_indices)

		# Getting the new learning rate
		if HALF_LEARNING_RATE_EVERY != 0:
			learning_rate_value = LEARNING_RATE * (0.5 ** epoch / HALF_LEARNING_RATE_EVERY)
		else:
			learning_rate_value = LEARNING_RATE

		#training step
		training_loss = 0.0

		for i in range(0, X_rgb_train.shape[0], BATCH_SIZE):
			indices = train_shuffled_indices[i:i + BATCH_SIZE]
			X_rgb_batch = X_rgb_train[indices]
			X_depth_batch = X_depth_train[indices]
			Y_batch = Y_train[indices]

			loss, _ = sess.run([cross_entropy_loss, train_op], 
								feed_dict = {rgb_input: X_rgb_batch, depth_input: X_depth_batch, correct_label: Y_batch,
											 rgb_keep_prob: KEEP_PROB, depth_keep_prob: KEEP_PROB, learning_rate: learning_rate_value})
			training_loss += loss

			iterations += 1
			
		# Evaluation step
		predictions, val_loss = predict(sess, rgb_input, depth_input, logits, rgb_keep_prob, depth_keep_prob, X_rgb_val, X_depth_val, Y_val, cross_entropy_loss, correct_label)
		iou = mean_iou(predictions, Y_val)

		training_loss /= X_rgb_train.shape[0]
		val_loss /= X_rgb_val.shape[0]

		print("Training averaged loss = {}".format(training_loss))
		print("Validation averaged loss = {}".format(val_loss))
		print("Validation mean_iou :", iou)

		# Writing the per epoch tensorboard things
		s = sess.run(epoch_tb_summ, feed_dict={training_loss_placeholder: training_loss, validation_loss_placeholder: val_loss, lr_placeholder: learning_rate_value, val_mean_iou_placeholder: iou})
		writer.add_summary(s, epoch)

		# Checkpoint by saving val predictions and the model
		if epoch % EPOCHS_BETWEEN_CHECKPOINTS == 0:
			path = OUT_DIR + 'epoch_' + str(epoch) + "/"

			if not os.path.exists(path):
				os.makedirs(path)

			save_predictions(X_rgb_val_orig, X_depth_val_orig, Y_val, predictions, path)

			# Save the model
			if val_loss <= best_val_loss:
				best_val_loss = val_loss

				saver.save(sess, OUT_DIR + 'model_checkpoint', global_step=epoch)

				print("Saved the model_checkpoint")


def predict(sess, rgb_image, depth_image, logits, rgb_keep_prob, depth_keep_prob, X_rgb, X_depth, Y, cross_entropy_loss, correct_label):

	# We need to do this in batches because it is too large
	predictions = []
	loss = 0.

	for i in range(0, X_rgb.shape[0], BATCH_SIZE):
		X_rgb_batch = X_rgb[i:i + BATCH_SIZE]
		X_depth_batch = X_depth[i:i + BATCH_SIZE]
		Y_batch = Y[i:i + BATCH_SIZE]

		batch_predictions, batch_loss = sess.run([logits, cross_entropy_loss], feed_dict={rgb_image: X_rgb_batch, depth_image: X_depth_batch, rgb_keep_prob: 1.0, depth_keep_prob: 1.0, correct_label: Y_batch})

		predictions.append(batch_predictions)
		loss += batch_loss

	predictions = np.concatenate(predictions)
	predictions = np.reshape(predictions, (X_rgb.shape[0], IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1))

	predictions = scipy.special.expit(predictions)

	return predictions, loss

def save_predictions(rgb_inputs, depth_inputs, outputs, predictions, path=OUT_DIR):
	for idx, prediction in enumerate(predictions):
		prediction = prediction[..., 0]
		img = Image.fromarray((prediction * 255).astype(np.uint8), 'L')
		img.save(path + str(idx) + '_prediction.png')

		img = Image.fromarray((outputs[idx][..., 0] * 255).astype(np.uint8), 'L')
		img.save(path + str(idx) + '_label.png')

		img = Image.fromarray(rgb_inputs[idx], 'RGB')
		img.save(path + str(idx) + '_rgb_input.png')

		img = Image.fromarray(depth_inputs[idx], 'RGB')
		img.save(path + str(idx) + '_depth_input.png')

	print('Saved {} prediction images'.format(len(predictions)))

def run():

	if not os.path.exists(OUT_DIR):
		os.makedirs(OUT_DIR)
	rgb_inputs, depth_inputs, outputs = load_datasets()
	
	X_rgb_train_orig, X_rgb_val_orig, X_depth_train_orig, X_depth_val_orig, Y_train_orig, Y_val_orig = train_test_split(rgb_inputs, depth_inputs, outputs, test_size=0.10, random_state=3)

	X_rgb_train, X_depth_train, Y_train = preprocess(X_rgb_train_orig, X_depth_train_orig, Y_train_orig)
	X_rgb_val, X_depth_val, Y_val = preprocess(X_rgb_val_orig, X_depth_val_orig, Y_val_orig)

	print("Data loaded and processed")

	with tf.Session() as session:

		# Tensorboard
		tf_writer = tf.summary.FileWriter(OUT_DIR, session.graph)

		# Returns the three layers, keep probability and input layer from the vgg architecture
		with tf.variable_scope("rgb_vgg"):
			rgb_input, rgb_keep_prob, rgb_layer3, rgb_layer4, rgb_layer7 = small_fcnn.load_vgg(session, VGG_PATH)
		with tf.variable_scope("depth_vgg"):
			depth_input, depth_keep_prob, depth_layer3, depth_layer4, depth_layer7 = small_fcnn.load_vgg(session, VGG_PATH)

		avg_layer3 = tf.math.add(rgb_layer3, depth_layer3)
		avg_layer3 = tf.math.scalar_mul(0.5, avg_layer3)
		avg_layer4 = tf.math.add(rgb_layer4, depth_layer4)
		avg_layer4 = tf.math.scalar_mul(0.5, avg_layer4)
		avg_layer7 = tf.math.add(rgb_layer7, depth_layer7)
		avg_layer7 = tf.math.scalar_mul(0.5, avg_layer7)

		correct_label = tf.placeholder(tf.float32, shape=(None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], OUTPUT_DIM))
		learning_rate = tf.placeholder(tf.float32)

		# The resulting network architecture from adding a decoder on top of the given vgg model
		model_output = small_fcnn.layers(avg_layer3, avg_layer4, avg_layer7)

		# Returns the output logits, training operation and cost operation to be used
		# - logits: each row represents a pixel, each column a class
		# - train_op: function used to get the right parameters to the model to correctly label the pixels
		# - cross_entropy_loss: function outputting the cost which we are minimizing, lower cost should yield higher accuracy
		logits, train_op, cross_entropy_loss = small_fcnn.optimize(model_output, correct_label, learning_rate)

		# Initialize all variables
		session.run(tf.global_variables_initializer())
		session.run(tf.local_variables_initializer())

		print("Model built successful, starting training")

		try:
			saver = tf.train.Saver(max_to_keep=1)
			saver.restore(session, OUT_DIR + 'model_checkpoint-70')
			train(session, train_op, cross_entropy_loss, rgb_input, depth_input, logits, correct_label, rgb_keep_prob, depth_keep_prob,
			 learning_rate, X_rgb_train, X_depth_train, Y_train, X_rgb_val, X_depth_val, Y_val, tf_writer, X_rgb_val_orig, X_depth_val_orig)
		finally:

			predictions, loss = predict(session, rgb_input, depth_input, logits, rgb_keep_prob, depth_keep_prob, X_rgb_val, X_depth_val, Y_val, cross_entropy_loss, correct_label)

			print("Final validation loss was {}".format(loss))
			save_predictions(X_rgb_val_orig, X_depth_val_orig, Y_val, predictions)

			tf.saved_model.simple_save(session, EXPORT_DIR, inputs={'rgbd_vgg/rgb_input:0': rgb_input, 'depth_vgg/depth_input:0': depth_input}, outputs={"fcn_logits": logits})

if __name__ == '__main__':
	run()