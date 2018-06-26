import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
########## Keras imports ###########
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Input, Activation, Reshape
from keras.models import Sequential, load_model, Model
from keras.layers.convolutional import Conv2D, ZeroPadding2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU, merge
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import Layer
import math
import tensorflow as tf
from PIL import Image
import sklearn
from sys import exit
import cv2
# Model architecture constants
# ===Base architectures===
SEGNET_ARCHITECTURE = 1
# ===Pretrained architectures===
VGG_NET = 2
INCEPTION_V3 = 3
RESNET = 4
INCEPTIONRESNET = 5
import tensorflow as tf
import utils


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


from keras import backend as K


class KerasModel:
    """ This class deals with creating, loading, and accessing Keras models.
        Here we'll deal with any operation that deals with keras this so we can have a
        clean and extensible code.
    """

    def __init__(self,
                 load=False,
                 model_file=None,
                 weighted=False):
        """ This method initializes the KerasModel object with the given flags
                architecture: constant int that specifies the model architecture, 1 by default
                load: bool that specifies if we should load model_file, False by default
                model_file: file to load model from, save trained model to, or both , None by default
                multivariant: bool which specifies if the model is multivariant, False by default
                gray: bool to indicate the use of Grayscaled images, False by default
        """
        if not load:
            # Create a Keras Sequential model as self.model
            # self.model = Sequential()
            # # Apply model preprocessing
            # self.model_preproceccing()
            # # Apply architecture
            self.segnet()
            # self.model = create_model()
        else:
            # Load the model
            if model_file:
                self.model = load_model(model_file)
                print('Succesfully loaded {}'.format(model_file))
            else:
                # Create a new model
                self.model = Sequential()
                print('No model to load, please specify a model_file')

    def load_model(model_file):
        """ This method loads a model of path model_file
                model_file: str of the model file path
        """
        # Loads a keras model
        self.model = load_model(model_file)
        print('Succesfully loaded {}'.format(model_file))

    def create_model(opt):
        """Create neural network model, defining layer architecture."""
        model = Sequential()
        # Convolution2D(output_depth, convolution height, convolution_width, ...)
        model.add(Convolution2D(20, 5, 5, border_mode='same',
                                input_shape=(256,
                                             256,
                                             3)))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(30, 5, 5, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(30, 5, 5, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(30, 5, 5, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(20, 5, 5, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(10, 5, 5, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(1, 5, 5, border_mode='same',
                                W_regularizer=l2(0.01), activation=tanh_zero_to_one))
        compile_model(model, opt)
        return model

    def segnet(self,
               kernel=3,
               n_classes=3,
               pool_size=(2, 2),
               input_shape=(256, 256, 3),
               pretrained_networks=VGG_NET):
        """ This method applies transfer learning to the model
            , its still a work in progress but works
                pretrained_networks: int flag specifying the pretrained network to choose from
        """
        # Data input
        # If you want to specify input tensor shape, e.g. 256x256 with 3 channels:
        input_tensor = Input(shape=input_shape)
        vgg_model = keras.applications.VGG16(weights='imagenet',
                                             include_top=False,
                                             input_tensor=input_tensor)

        # To see the models' architecture and layer names, run the following
        # vgg_model.summary()
        # exit()
        # model.add(InputLayer(input_tensor=tf.nn.fractional_max_pool(model.layers[3].output, p_ratio)[0]))
        # Decoder Layers
        decoder = vgg_model.output
        # DeConv 1
        # self.model.add(MaxUnpooling2D(pool_size))
        decoder = UpSampling2D(size=pool_size)(decoder)
        decoder = Conv2D(512, (kernel, kernel), padding='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Activation('relu')(decoder)
        decoder = Conv2D(512, (kernel, kernel), padding='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Activation('relu')(decoder)
        decoder = Conv2D(512, (kernel, kernel), padding='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Activation('relu')(decoder)

        # DeConv 2
        # self.model.add(MaxUnpooling2D(pool_size))
        decoder = UpSampling2D(size=pool_size)(decoder)
        decoder = Conv2D(512, (kernel, kernel), padding='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Activation('relu')(decoder)
        decoder = Conv2D(512, (kernel, kernel), padding='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Activation('relu')(decoder)
        decoder = Conv2D(256, (kernel, kernel), padding='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Activation('relu')(decoder)

        # DeConv 3
        # self.model.add(MaxUnpooling2D(pool_size))
        decoder = UpSampling2D(size=pool_size)(decoder)
        # decoder = ZeroPadding2D((1, 2))(decoder)
        decoder = Conv2D(256, (kernel, kernel), padding='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Activation('relu')(decoder)
        decoder = Conv2D(256, (kernel, kernel), padding='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Activation('relu')(decoder)
        decoder = Conv2D(128, (kernel, kernel), padding='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Activation('relu')(decoder)

        # DeConv 4
        # self.model.add(MaxUnpooling2D(pool_size))

        decoder = UpSampling2D(size=pool_size)(decoder)
        # decoder = ZeroPadding2D((3, 4))(decoder)
        decoder = Conv2D(128, (kernel, kernel), padding='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Activation('relu')(decoder)
        decoder = Conv2D(64, (kernel, kernel), padding='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Activation('relu')(decoder)
        # DeConv 5
        # self.model.add(MaxUnpooling2D(pool_size))
        decoder = UpSampling2D(size=pool_size)(decoder)
        decoder = Conv2D(64, (kernel, kernel), padding='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Activation('relu')(decoder)
        decoder = Conv2D(n_classes, (1, 1), padding='valid')(decoder)
        decoder = BatchNormalization()(decoder)

        # decoder = Reshape((224 * 512, 3))(decoder)

        predictions = Activation('softmax')(decoder)

        # this is the model we will train
        self.model = Model(inputs=vgg_model.input, outputs=predictions)
        # self.model.summary()
        # Freeze all pretrained Layers weights and biases
        # self.freeze_model_layers(base_model)
        for layer in vgg_model.layers:
            layer.trainable = False
        # self.model.summary()
        # exit()

    # def process_prediction(self, softmax_image, original_image):
    #     image_new = np.zeros_like(original_image)

    #     reshaped_softmax = softmax_image.reshape(-1, softmax_image.shape[-1])
    #     max_channel = np.argmax(reshaped_softmax, -1).reshape((256, 256))
    #     road_pixels = (max_channel[:, :] == 0).nonzero()
    #     vehicle_pixels = (max_channel[:, :] == 1).nonzero()

    #     image_new[:, :, 0][road_pixels] = 1
    #     image_new[:, :, 1][vehicle_pixels] = 1
    #     return cv2.resize(image_new, (1280, 720))

    def process_prediction(self, softmax_image, original_image, argmax = True):
        image_new = np.zeros_like(original_image)
        confidence = 0.35
        
        if argmax:
            reshaped_softmax = softmax_image.reshape(-1, softmax_image.shape[-1])
            max_channel = np.argmax(reshaped_softmax, -1).reshape((256, 256))
            road_pixels = (max_channel[:, :] == 0).nonzero()
            car_pixels = (max_channel[:, :] == 1).nonzero()
        else:
            road_pixels = (softmax_image[:, :, 0] > confidence).nonzero()

        image_new[:, :, 2][road_pixels] = 1
        image_new[:, :, 1][car_pixels] = 1
        
        return cv2.resize(image_new, (1280, 720))

    def predict(self, image):
        preprocessed_image = preprocess_image(image)
        prediction = self.model.predict(preprocessed_image[None, :, :, :])[0]
        lane_image = self.process_prediction(prediction, preprocessed_image)
        return lane_image

    def freeze_model_layers(self, model):
        """ This method  freezes a given model layers weights for feature extraction
                model: the model to freeze its weights
        """
        for layer in model.layers:
            # Sets the layer's weights to be un-trainable
            layer.trainable = False

    def unet(self, input_size=(256, 256, 3)):
        inputs = Input(input_size)
        # Normalizing and standardizing our images
        # inputs = Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_size)
        conv1 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same',
                     kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same',
                     kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same',
                     kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same',
                     kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(16, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(3, 1, activation='sigmoid')(conv9)

        self.model = Model(input=inputs, output=conv10)
        # self.model.summary()

    def train_model_with_generator(self,
                                   train_generator,
                                   steps_per_epoch,
                                   epochs,
                                   validation_generator=None,
                                   validation_steps=None,
                                   save_model_filepath='model.h5'):
        """ This method defines the model training configuration
            via calling the Keras model.compile() method
            that takes the loss function and optimizer type, then
            it calls model.fit_generator() to train the network on the
            given generators. The method also keeps track of the model training metrics
            in model_history then returns it for further analysis
                train_generator: Training data Python generator
                steps_per_epoch: int number of batches that the fit_generator() method will accept before declaring on epoch
                epochs: int number of epochs to train
                validation_generator: Validation data Python generator, with a default argument of None
                validation_steps: int Number of batches that the fit_generator() method will accept before declaring on epoch
                                  , with a default argument of None
                save_model_filepath: h5 model file path to save to , with a default argument of 'model.h5'
        """
        # run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        # Defining the loss function and optimizer
        # sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4),
                           metrics=['accuracy'])  # , options=run_opts)
        # Early stopping callback
        earlyStoppingCallBack = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              patience=30,
                                                              verbose=0,
                                                              mode='auto')
        # Creates a checkpoint and saves it if the val_loss decreased
        checkpointer = ModelCheckpoint(filepath='tmp/best_model.h5',
                                       verbose=1, save_best_only=True)
        # Tensorboard callback
        tbCallBack = keras.callbacks.TensorBoard(
            log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        # Training summary
        print('Started training on ')
        print('Data')

        # keras.utils.print_summary(self.model, line_length=None, positions=None, print_fn=None)
        # Training the model and getting the model history for future visualization
        model_history = self.model.fit_generator(train_generator,
                                                 steps_per_epoch=steps_per_epoch,
                                                 validation_data=validation_generator,
                                                 validation_steps=validation_steps,
                                                 epochs=epochs,
                                                 verbose=1,
                                                 callbacks=[tbCallBack,
                                                            checkpointer,
                                                            earlyStoppingCallBack])

        # Saving the model to the save file
        self.model.save(save_model_filepath)
        print('Saved model to {}'.format(save_model_filepath))
        # Returning the model history for showing loss graph
        return model_history


def augment_data(image, label_image, augment=False):
    """ This method apply augmentation on the given images and measurements.
            images: the input images array to augment
            mesurements: the input mesurements array to augments
            augment: a bool to apply all augmentation mainly shadow, False by default
            multivariant: a bool to specify if the data is multivariant, False by default
    """
    # Initializing empty augmented arrays
    augmented_images = []
    augmented_label_images = []
    # Applying augmentation on a fraction of the images
    # n_images_to_augment = int(len(images) * 0.30)
    # For every image and its steering angle flip the image and apply augmentation on shuffled
    # fraction of the images

    # augment is only true in training
    # if augment and n_images_to_augment > 0:

    # Create a flipped version of the image and steering angle
    flipped_image, flipped_label_image = flip_images(image,
                                                     label_image)
    augmented_images.append(flipped_image)
    augmented_images.append(image)
    augmented_label_images.append(flipped_label_image)
    augmented_label_images.append(label_image)

    augmented_images + rotate_image(image)
    augmented_label_images + rotate_image(label_image)
    return augmented_images, augmented_label_images


def flip_images(image, label_image):
    """ This method creates a flipped version of the input image and steering angle
            image: input image to flip
            steering_angle: either a steering angle or a steering angel and a throttle value
                            if multivariant is true
            multivariant: bool that specifies if the steering angle has multiple values, False by
                          default
    """
    flipped_image = np.fliplr(image)
    flipped_label_image = np.fliplr(label_image)
    return flipped_image, flipped_label_image

def rotate_image(img):
    pil_img = Image.fromarray(img.astype('uint8'), 'RGB') 
    return [np.array(pil_img.rotate(25)),
           np.array(pil_img.rotate(-25)), 
           np.array(pil_img.rotate(45)),
           np.array(pil_img.rotate(-45))]


def get_images_from_paths(img_path, label_path):
    """ This method loads images and gets mesurement values from a str array input line
            lines: array of str arrays of data which are a line in a csv document
            multivariant: a bool to indicate wether or not its a multivariant model, False by default
        returns
    """
    # image_dir = "Train/CameraRGB/"
    # label_image_dir = "Train/CameraSeg/"
    image = np.array(Image.open(img_path))  # cv2.imread(img_path)
    label_image = np.array(Image.open(label_path))  # cv2.imread(img_path)
    #  Preprocess label image
    label_image = preprocess_image_labels(label_image)
    # print('image type {}'.format(label_image.dtype))
    # print(img_path)
    return preprocess_image(image), label_image


# def preprocess_image_labels(label_image):
#     LANE_LABEL = 34
#     CAR_LABEL = 26
#     # SIGN_LABEL = 0.078431375
# #     labels_new = np.zeros((label_image.shape[0], label_image.shape[1], 4))
#     labels_new = np.zeros_like(label_image)

#     # Identify lane marking pixels (label is 6)
#     lane_marking_pixels = (label_image[:, :, 0] == LANE_LABEL).nonzero()
#     labels_new[:, :, 0][lane_marking_pixels] = 1

#     # Identify all vehicle pixels
#     vehicle_pixels = (label_image[:, :, 0] == CAR_LABEL).nonzero()
#     # Set Vehicles pixels
#     labels_new[:, :, 1][vehicle_pixels] = 1

#     # Identify all sign pixels
# #     sign_pixels = (label_image[:, :, 0] == SIGN_LABEL).nonzero()
#     # Set Vehicles pixels
# #     labels_new[:, :, 2][sign_pixels] = 1

#     # Find all other labels
#     other_pixels = ((label_image[:, :, 0] != LANE_LABEL)
#                     & (label_image[:, :, 0] != CAR_LABEL))
#     # Remove the labels by setting their pixels to 0 ~ None
#     labels_new[:, :, 2][other_pixels] = 1
#     return cv2.resize(labels_new, (256, 256))


# def preprocess_image_labels(label_image):
#     LANE_LABEL = 229
#     labels_new = np.zeros_like(label_image[:, :, :2])

#     # Identify lane marking pixels (label is 6)
#     lane_marking_pixels = (label_image[:, :, 0]).nonzero()
#     labels_new[:, :, 0][lane_marking_pixels] = 1


#     # Find all other labels
#     other_pixels = ((label_image[:, :, 0] != LANE_LABEL))
#     # Remove the labels by setting their pixels to 0 ~ None
#     labels_new[:, :, 1][other_pixels] = 1
#     return cv2.resize(labels_new, (256, 256))


def preprocess_image_labels(label_image):
    # LANE_LABEL = 255
    # CAR_LABEL = 142
    # labels_new = np.zeros_like(label_image[:, :, :])

    # # Identify lane marking pixels (label is 255)
    # lane_marking_pixels = (label_image[:, :, 0]).nonzero()
    # labels_new[:, :, 0][lane_marking_pixels] = 1

    # # Identify car pixels (label is 142)
    # car_pixels = (seg_image[:, :, 2] == CAR_LABEL).nonzero()
    # labels_new[:, :, 1][car_pixels] = 1


    # # Find all other labels
    # other_pixels = ((label_image[:, :, 0] != LANE_LABEL) 
    #                 & (seg_image[:, :, 2] != CAR_LABEL))
    # # Remove the labels by setting their pixels to 0 ~ None
    # labels_new[:, :, 2][other_pixels] = 1
    return cv2.resize(label_image, (256, 256))


def preprocess_image(image):
    return cv2.resize(image, (256, 256)) / 255.0 - 0.5


def data_generator1(data_element, batch_size):
    """ This method is a Python Generator that takes in a number of samples (lines) and yeilds
        loaded augmented batch sized images and mesurements.
            samples: Array of str arrays (lines) each contain csv line of data
            batch_size: int number of data points to yield for one batch
            validation: Bool to specify whether or not the generator is for trainig or validation, False by default
            multivariant: Bool to specify whether or not the data should be multivariant, False by default
        yields batch_size images and mesurements
    """
    while True:  # Forever loop to keep the generator up till the termination of the program
             # (end of training and validation)
        # shuffling input samples for good measure
        data_element.shuffle_data()
        # Empty arrays for data collection
        X_data = []
        y_data = []
        # exit()
        # Iterates for every sample
        for i, (feature_path, label_path) in enumerate(zip(data_element.X, data_element.y)):
                # Get the samples images, which will return 3 images (center, left, right)
            # and their angles
            image, label_image = get_images_from_paths(feature_path, label_path)
            # print('salsdjkfj{}'.format(image.shape))
            # Augment sample images flip, but adds shadow to only training data
            augmented_images, augmented_label_images = augment_data(image,
                                                                    label_image,
                                                                    augment=True)
            # Adding our generated sample data into our yield arrays
            X_data.extend(augmented_images)
            y_data.extend(augmented_label_images)
            # Check if X is of batch_size or if its the last element
            # Yield if we have collected a batch_size or more (due to concurrent loading) or if its
            # the last batch which will usually be less than batch_size
            if len(X_data) > batch_size or i == data_element.len - 1:
                # print('==================Batch====================')
                # print('At count: {}'.format(i))
                # Putting our augmented data into numpy arrays cause Keras require numpy arrays
                # yield the batch
                yield sklearn.utils.shuffle(np.array(X_data[:batch_size]), np.array(y_data[:batch_size]))
                # Keep any extra data that was loaded but exceded the batch_size for next batch
                X_data = X_data[batch_size:]
                y_data = y_data[batch_size:]


def get_data_generator_and_steps_per_epoch(data_element, batch_size, validation=False, weighted=False, augment=True):
    """ This method creates a generator and calculates the steps_per_epoch for the generator based
        on images loaded and augmentation applied.
            samples: Array of str arrays (lines) each contain csv line of data
            batch_size: int number of data points to yield for one batch
            validation: Bool to specify whether or not the generator is for trainig or validation, False by default
            multivariant: Bool to specify whether or not the data should be multivariant, False by default
        returns a generator and steps_per_epoch
        """
    # Constant of number of augmentation
    N_AUGMENTATION = 4 + 1 + 1 if augment else 1
    # A generator for the samples given be it a training, validation, or test samples
    generator = data_generator1(data_element, batch_size)
    # Calculates the number of images shadow augmentation adds to the data
    # shadow_augmentation = int(len(samples) * 3 * 0.3) if not validation else 0
    # Number of batches that the fit_generator() method will accept before declaring on epoch
    print((data_element.len * N_AUGMENTATION))
    steps_per_epoch = math.ceil(
        ((data_element.len * N_AUGMENTATION)) / batch_size)
    return generator, steps_per_epoch
