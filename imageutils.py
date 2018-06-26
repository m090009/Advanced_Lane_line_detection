import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import datetime


# ___________________ Image Convertions_________________________


def convert_to_gray(img, expand=False):
    image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return image if not expand else np.expand_dims(image, axis=-1)


def convert_to_hsl(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

# ______________________ Image Transformations__________________


def region_of_interest(img, vertices):
    """ Applies a quadratic mask of vertices,
        and only shows the pixels inside that mask.abs
            img: image to apply mask to.
            vertices: a set of 4 points to define the mask.
    """
    # Empty array with img size and pixels set to zeros
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
