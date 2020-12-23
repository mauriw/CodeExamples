"""
Currently configured for the salnet model

"""
from PIL import Image
import glob
import Augmentor
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import cv2
from shutil import rmtree

DATA_DIRECTORY = 'data/larger/'
OUTPUT_DIRECTORY = 'data/augmented/'

def generateInputs(rgb, depth, suction):
	"""
	takes in list of filenames
	generates the input in format for the DataPipeline
	"""
	zipped = list(zip(rgb, depth, suction))
	inputs = [[np.asarray(Image.open(y)) for y in x] for x in zipped]
	return inputs

def getDataNames(directory):
	"""
	iterates through the directory and returns an rgb, suction, and depth list
	each containing the filenames of its respective images
	"""
	rgb, depth, suction = [], [], []

	for filename in glob.glob(directory + '*rgb.png'):
		file_id = filename[len(DATA_DIRECTORY): -8]
		rgb.append(directory + file_id + '-rgb.png')
		depth.append(directory + file_id + '-depth.png')
		suction.append(directory + file_id + '-suction.png')

	return rgb, depth, suction

def process_depth(depth_path, size):
	"""
	arguments: a path to a suction image and size to resize to
	does: converts the suction image to depth jet and resizes it
	returns: processed CV2 image

	# min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(depth_image)
    # depthJet_image = cv2.convertScaleAbs(depth_image, None, 255 / (max_val-min_val), -min_val);
	"""
	img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
	jet = cv2.applyColorMap(img, cv2.COLORMAP_JET)
	final = cv2.resize(jet, size)
	return final

def resize_and_save(rgbs, depths, suctions, size):
	"""
	arguments: lists of filepaths to original images
	resizes all images and saves them to OUTPUT_DIRECTORY
	"""
	for i in range(len(rgbs)):
		rgb = Image.open(rgbs[i]).resize(size)
		depth = process_depth(depths[i], size)
		suction = Image.open(suctions[i]).resize(size).convert('L')
		save(rgb, None, suction, i)
		saveDepth(depth, i)

def save_augmented(images, start):
	for i in range(len(images)):
		rgb = Image.fromarray(images[i][0])
		depth = Image.fromarray(images[i][1])
		suction = Image.fromarray(images[i][2])
		save(rgb, depth, suction, start + i)

def saveDepth(depth_img, index):
	"""
	given a CV2-formatted depth image, saves it with indexed name to OUTPUT directory
	"""
	fName = OUTPUT_DIRECTORY + str(index) + '-depth.png'
	cv2.imwrite(fName, depth_img)

def save(rgb, depth, suction, index):
	"""
	given 3 Pillow images saves them with an indexed name to the OUTPUT directory
	"""
	rgbFName = OUTPUT_DIRECTORY + str(index) + '-rgb.png'
	rgb.save(rgbFName)

	if depth:
		depthFName = OUTPUT_DIRECTORY + str(index) + '-depth.png'
		depth.save(depthFName)

	suctionFName = OUTPUT_DIRECTORY + str(index) + '-suction.png'
	suction.save(suctionFName)

def showRandomInput(images, labels):
	r_index = random.randint(0, len(images)-1)
	f, axarr = plt.subplots(1, 3, figsize=(6,2))
	axarr[0].imshow(images[r_index][0])
	axarr[1].imshow(images[r_index][1], cmap="gray")
	axarr[2].imshow(labels[r_index], cmap="gray")
	plt.show()

#the parameters for these are fairly arbitrary, feel free to change them up to add more or less variety
def augment_operations(p):
	p.rotate(1, max_left_rotation=5, max_right_rotation=5)
	p.flip_left_right(probability=0.5)
	p.zoom_random(probability=0.8, percentage_area=0.8)
	p.skew(probability=.3, magnitude=.5)
	p.shear(probability=.3, max_shear_left=5, max_shear_right=5)

def checkOutputDirectory():
	if not os.path.exists(OUTPUT_DIRECTORY):
		print("Creating output directory")
		os.makedirs(OUTPUT_DIRECTORY)

def removeOutputDir():
	if os.path.exists(OUTPUT_DIRECTORY):
		rmtree(OUTPUT_DIRECTORY)

def main():
	IMAGE_SIZE = (224, 224)
	NUM_SAMPLES = 0

	print('Looking for images in {}'.format(DATA_DIRECTORY))
	rgb, depth, suction = getDataNames(DATA_DIRECTORY)
	print('Found {} rgb images'.format(len(rgb)))
	print('Found {} depth images'.format(len(depth)))
	print('Found {} suction images'.format(len(suction)))

	print()
	print('Removing everything previously in {}'.format(OUTPUT_DIRECTORY))
	removeOutputDir()

	print("Creating new output directory {}".format(OUTPUT_DIRECTORY))
	checkOutputDirectory()

	print()
	print('Resizing all images to size {}'.format(IMAGE_SIZE))
	print('Converting all suction images to depth jet color scheme')
	print('Grayscaling all suction images')
	resize_and_save(rgb, depth, suction, IMAGE_SIZE)


	# print()
	# print('Creating {} augumented examples'.format(NUM_SAMPLES))
	# rgb, depth, suction = getDataNames(OUTPUT_DIRECTORY) #work with the resized images
	# images = generateInputs(rgb, depth, suction)
	# p = Augmentor.DataPipeline(images)
	# augment_operations(p)
	# aug_images = p.sample(NUM_SAMPLES)

	# # showRandomInput(aug_images, aug_labels)
	# save_augmented(aug_images, len(rgb))
	print('Saved resized and augumented examples to {}'.format(OUTPUT_DIRECTORY))

if __name__ == '__main__':
    main()