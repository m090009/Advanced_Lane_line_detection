import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import datetime
import os
import pickle
import math
CALIBRATION_PICKLE_FILE = 'calibration_points.p'


def plot_loss(model_history):
    print(model_history.history.keys())
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def Load_images_for_directory(directory):
    return [mpimg.imread(directory + image_name) for image_name in os.listdir(directory)]


def show_images(images, image_name='', cmap=None, cols=2):
    """ Shows images in a grid and saves it as image_name
            images: dfsgdfg
            image_name: fdgdfg
            cmap:dfggdf g
            cols:dfgdfg
    """
    SAVE_DIR = 'output_images/'
    directory = ''
    cols = cols
    rows = (len(images)+1)//cols

    plt.figure(figsize=(10, 11))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        # use gray scale color map if there is only one channel
        cmap = 'gray' if len(image.shape) == 2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    if os.path.isdir(SAVE_DIR):
        directory = SAVE_DIR
    image_name = str(datetime.datetime.now()).split('.')[0].replace(' ', '').replace(
        ':', '').replace('-', '') if image_name == '' else image_name
    plt.savefig(directory + image_name + '.png', bbox_inches='tight')
    plt.show()


def wrap_words(line, wrap=5):
    if len(line.split(' ')) >= wrap:
        line = line.split(' ')
        line.insert(wrap, "\n")
        return' '.join(line)
    else:
        return line


def show_images(images,
                images_titles=[],
                image_name='',
                cmap=None,
                save=True,
                horizontal=False,
                cols=4):
    SAVE_DIR = 'images_output/'
    directory = ''
    # if horizontal:
    cols = cols
    rows = int(math.ceil(len(images) / cols))
    # else:
    #     rows = 8
    #     cols = int(math.ceil(len(images) / rows))
    plt.figure(figsize=(15, 15))
    # Wrapping all titles to two lines if they have more than 4 words using the wrap_words method
    images_titles = list(map(wrap_words, images_titles))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        # use gray scale color map if there is only one channel
        if len(image.shape) == 3 and image.shape[2] == 1:
            cmap = 'gray'
            image = image.squeeze()
        if len(image.shape) == 2:
            cmap = 'gray'

        plt.imshow(image, cmap=cmap)
        if len(images_titles) == len(images):
            plt.title(images_titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    if save:
        if os.path.isdir(SAVE_DIR):
            directory = SAVE_DIR
        image_name = str(datetime.datetime.now()).split('.')[0].replace(' ', '').replace(
            ':', '').replace('-', '') if image_name == '' else image_name
        plt.savefig(directory + image_name + '.png', bbox_inches='tight')
        # plt.tight_layout()
    plt.show()


def save_calibration_points_to_pickle(object_points, image_points):
    """ Saves calibration points to pickle file CALIBRATION_PICKLE_FILE
            object_points:
            image_points:
    """
    calibration_points = {'object_points': object_points, 'image_points': image_points}
    pickle.dump(calibration_points, open(CALIBRATION_PICKLE_FILE, 'wb'))
    print('Saved Calibration points to {}'.format(CALIBRATION_PICKLE_FILE))


def load_calibration_points_from_pickle():
    """ Loads calibration points from pickle file CALIBRATION_PICKLE_FILE
        Returns: a dictionary of the calibration points
    """
    print('Loaded Calibration points from {}'.format(CALIBRATION_PICKLE_FILE))
    return pickle.load(open(CALIBRATION_PICKLE_FILE, 'rb'))
