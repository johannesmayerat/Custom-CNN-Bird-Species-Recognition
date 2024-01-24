#
# Custom CNN for classifying 400 bird species using Keras
# Johannes Mayer
# hnsmyr@gmail.com
# 
# NOTE:
#  * For the current setup, about 80GB RAM are needed. If not available, reduce N_SPEC, N_IMG_PER_SPECIES or SEL_* (data augmentation)
#  * Corresponding dataset can be found here: https://www.kaggle.com/datasets/gpiosenka/100-bird-species
#  * This model was created in June 2022 (only 400 bird species were available from the given dataset)
#

import os
import numpy as np
import pandas as pd 
import datetime
import matplotlib.pyplot as plt
import matplotlib.image as image

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
print('Start time: ',datetime.datetime.now().time())

# Path of the bird species dataset, must contain the file birds.csv
# see https://www.kaggle.com/datasets/gpiosenka/100-bird-species
ipath = './archive-data/' 

n_epochs = 36
N_SPECIES = 400
N_IMG_PER_SPECIES = 120
sel_size = (112,112)
SEL_LR_FLIP=True
SEL_ROTATE=True
SEL_ZOOM=True
OPTIMIZER='adam'

# This function initializes the CNN model.
def initialize_model(name):
    model = Sequential(name=name)

    model.add(layers.Conv2D(32, (7, 7), activation="relu", input_shape=img_size, padding='same')) # dtype=tf.bfloat16,
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3, 3)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding='same')) # dtype=tf.bfloat16,
    model.add(layers.BatchNormalization()) 

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding='same')) # dtype=tf.bfloat16,
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding='same')) # dtype=tf.bfloat16, 
    model.add(layers.BatchNormalization())  

    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding='same')) # dtype=tf.bfloat16,
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding='same')) # dtype=tf.bfloat16,
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    
    model.add(layers.Dropout(rate=0.2))  
    model.add(layers.Dense(512, activation='relu')) # dtype=tf.bfloat16,

    model.add(layers.Dropout(rate=0.3)) 
    model.add(layers.Dense(512, activation='relu')) # dtype=tf.bfloat16,
    
    model.add(layers.Dropout(rate=0.5)) 
    model.add(layers.Dense(512, activation='relu')) # dtype=tf.bfloat16,

    model.add(layers.Dense(N_SPECIES, activation='softmax'))

    return model


# This function reads the images and applies some data augmentation operations, such as rotation, zoom, flip.
def ReadImages(data_type='train', n_species=50, n_img_per_species=120,\
               sel_lr_flip=False, sel_rotate=False, sel_zoom=False, random_shuffle=True,\
               img_size=(224,224)):
    
	n_total = n_species*n_img_per_species
	n_add = 1

	if sel_lr_flip: 
		n_add += 1
		
	if sel_rotate: 
		n_add += 1
		if sel_lr_flip:
			n_add += 1
		
	if sel_zoom: 
		n_add += 1  	
		if sel_lr_flip: 
			n_add += 1


	df = pd.read_csv(ipath+'birds.csv')
	df = df[df['data set'] == data_type]
	df = df[df['class index'] < n_species]
	df = df.groupby('labels').head(n_img_per_species)

	labels_ini = df['class index'].values

	data = np.zeros((n_total*n_add,)+img_size+(3,))
	labels = np.zeros([n_total*n_add])

	filepaths = list(df['filepaths'])
	print(f' -- Reading {n_total} images (creating {n_add-1} copies).')

	for i in range(n_total):
		if i%1000 == 0: print(f' Images read: {i}')


		label_this_img = labels_ini[i]

		ipath_img = ipath + filepaths[i]
		#print(ipath_img)
		tmp_img =  np.flip(image.imread(ipath_img),axis=0)/255

		if img_size != (224,224): tmp_img = np.array(tf.image.resize(tmp_img,img_size))

		data[i,:,:,:] = tmp_img[:,:,:]
		labels[i] = label_this_img

		jumper = 0
		if sel_lr_flip:
			jumper += n_total
			tmp_img_flip = np.array(tf.image.flip_left_right(tmp_img))
			data[jumper+i,:,:,:] = tmp_img_flip[:,:,:]
			labels[jumper+i] = label_this_img
		
		if sel_rotate:
			jumper += n_total
			tmp_img_rot = tf.keras.preprocessing.image.random_rotation(tmp_img,30,row_axis=0,col_axis=1,channel_axis=2)
			data[jumper+i,:,:,:] = tmp_img_rot[:,:,:]
			labels[jumper+i] = label_this_img
		
			if sel_lr_flip:
				jumper += n_total
				tmp_img_rot = tf.keras.preprocessing.image.random_rotation(tmp_img_flip,30,row_axis=0,col_axis=1,channel_axis=2)
				data[jumper+i,:,:,:] = tmp_img_rot[:,:,:]
				labels[jumper+i] = label_this_img

		if sel_zoom:
			jumper += n_total
			tmp_img_zoom = tf.keras.preprocessing.image.random_zoom(tmp_img,(0.9,0.9),row_axis=0,col_axis=1,channel_axis=2)
			data[jumper+i,:,:,:] = tmp_img_zoom[:,:,:]
			labels[jumper+i] = label_this_img
			
			if sel_lr_flip:
				jumper += n_total
				tmp_img_zoom = tf.keras.preprocessing.image.random_rotation(tmp_img_flip,30,row_axis=0,col_axis=1,channel_axis=2)
				data[jumper+i,:,:,:] = tmp_img_zoom[:,:,:]
				labels[jumper+i] = label_this_img


	if random_shuffle:
		idx = np.random.permutation(n_total*n_add)
		labels = labels[idx]
		data = data[idx,:,:,:]

	labels = to_categorical(labels)

	return data, labels, n_total


## Check if species of validation set are actually in training set
def check_same_labels(labels1,labels2):
	if len(np.unique(labels1)) != len(np.unique(labels2)):
		print(f'Incorrect label length: {len(np.unique(labels1))} != {len(np.unique(labels2))}')
		exit()

	cnt = 0
	for i in range(len(labels1)):
		if labels1[i] not in labels2: 
			print(labels1[i])
			cnt += 1

	if cnt != 0: exit()

def plot_img(data):
	plt.figure(figsize=(7,7))
	ax = plt.subplot()
	ax.imshow(np.flip(data,axis=0))
	plt.show() 

def print_setup():
	print(f'          Optimizer = {OPTIMIZER}')
	print(f'  Number of species = {N_SPECIES}')
	print(f' Images per species = {N_IMG_PER_SPECIES}')
	print(f'         Image size = ', sel_size)
	print(f'               Flip = {SEL_LR_FLIP}')
	print(f'             Rotate = {SEL_ROTATE}')
	print(f'               Zoom = {SEL_ZOOM}')
	print(f'   Number of epochs = {n_epochs}')


if __name__ == "__main__":

	print(' -- Read training data')
	data_train, labels_train, n_total_train = ReadImages(n_species=N_SPECIES, n_img_per_species=N_IMG_PER_SPECIES,\
								img_size=sel_size, sel_lr_flip=SEL_LR_FLIP, sel_rotate=SEL_ROTATE, sel_zoom=SEL_ZOOM)
	img_size = data_train[0].shape
	
	print(' -- Read validation data')
	data_val, labels_val, n_total_val = ReadImages(data_type='valid',n_species=N_SPECIES, n_img_per_species=5,img_size=sel_size)

	print(' -- Read test data')
	data_test, labels_test, n_total_test = ReadImages(data_type='test',n_species=N_SPECIES, n_img_per_species=5,img_size=sel_size)

	print(' -- Initialize and compile model')
	model = initialize_model(name="baseline")
	model.summary()
	model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics="accuracy")

	callback = [EarlyStopping(patience=4, monitor='val_accuracy', restore_best_weights=True),\
		    ReduceLROnPlateau(monitor = 'val_loss', patience = 2, factor=0.5, verbose=1)]

	print_setup()

	print(' -- Train neural network')
	history_model = model.fit(data_train, labels_train, batch_size=32, epochs=n_epochs, validation_data=(data_val, labels_val), callbacks=callback)

	print(' -- Evaluate model')
	model.evaluate(data_test,labels_test)

	model_filename = f'CNNv6-6xCONV-4xDENSE-{OPTIMIZER}-{N_SPECIES}-{N_IMG_PER_SPECIES}-{sel_size[0]}-{n_epochs}-{SEL_LR_FLIP}-{SEL_ROTATE}-{SEL_ZOOM}'

	model.save(model_filename)

	print(f'End time: {datetime.datetime.now().time()}')
	print(' *** DONE *** ')



