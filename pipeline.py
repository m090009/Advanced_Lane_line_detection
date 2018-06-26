import os 
import utils
import imageutils
import adv_laneline_detection
import matplotlib.image as mpimg
from importlib import reload
import matplotlib.pyplot as plt
import numpy as np
import cv2
import kerasmodel
import datasetclasses
from PIL import Image




training_dir = 'bb/train/'
labels_dir = 'bb/lanelines_labels/'

training_images_paths = [(training_dir + image_name) for image_name in os.listdir(training_dir)]
label_images_paths = [(labels_dir + image_name) for image_name in os.listdir(labels_dir)]
dataset = datasetclasses.Dataset(training_images_paths, label_images_paths)
print('Training on {} images'.format(dataset.train.len))
print('Validating on {} images'.format(dataset.valid.len))
print(training_images_paths[0])
# utils.show_images(label_images)
BATCHSIZE = 8

print('Training generator')
train_generator, train_steps_per_epoch = kerasmodel.get_data_generator_and_steps_per_epoch(dataset.train,
                                                                            BATCHSIZE)

print('Validation generator')
validation_generator, validation_steps_per_epoch = kerasmodel.get_data_generator_and_steps_per_epoch(dataset.valid,
                                                                                      BATCHSIZE,
                                                                                      validation=True)

print('Training steps per epoch {}'.format(train_steps_per_epoch))
print('Validation steps per epoch {}'.format(validation_steps_per_epoch))


model_file = 'model_berkely.h5'
k_model = kerasmodel.KerasModel(model_file=model_file,
                             load=False)
EPOCHS = 20
# k_model.model.summary()
# Training the KerasModel model and getting the metrics
model_history = k_model.train_model_with_generator(train_generator,
                                                   train_steps_per_epoch,
                                                   EPOCHS,
                                                   validation_generator,
                                                   validation_steps_per_epoch,
                                                   save_model_filepath=model_file)
# Plotting the model Loss
utils.plot_loss(model_history=model_history)