import utils
import imageutils
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import kerasmodel

# load in Chessboard Calibration images
directory = "camera_cal/"
project_test_images = utils.Load_images_for_directory(directory)


def get_image_corners(img, nx, ny):
    """ Gets the image corners for img

    """
    # nx, ny = 9, 6
    # Convert image to grayscale
    gray = imageutils.convert_to_gray(img)
    # Get Chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    return ret, corners


def get_image_points(img, nx=9, ny=6):
    """ Gets image points and object points for each image in imgs

        Returns:
    """
    # Prepare object point by creating a zeros array of the same size as the image
    object_points = np.zeros((nx*ny, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    ret, corners = get_image_corners(img, nx, ny)
    if ret:
        # Add image corners
        image_points = corners
        # Create an image copy to draw on
        image_copy = img.copy()
        # Draw found corners on the image
        corner_image = cv2.drawChessboardCorners(image_copy,
                                                 (nx, ny),
                                                 corners,
                                                 ret)
        return object_points, image_points, corner_image
    else:
        return np.array([]), np.array([]), np.array([])


def get_images_points(imgs, nx=9, ny=6):
    """ Gets image points and object points for each image in imgs

        Returns:
    """
    images_object_points = []
    images_points = []
    corner_images = []

    for img in imgs:
        object_points, image_points, corner_image = get_image_points(img, nx, ny)
        if object_points.size > 0:
            # Add image corners
            images_points.append(image_points)
            # Add the prepared object points
            images_object_points.append(object_points)
            # Draw found corners on the image
            corner_images.append(corner_image)

    return images_object_points, images_points, corner_images


def calibrate_camera(images_object_points, images_points, images_shape):
    """ Calibrates camera images

    """
    # Calibrate camera using found corners
    # image_shape = img.shape[1::-1]
    ret, mtx, dist, rvecs, tvec = cv2.calibrateCamera(images_object_points,
                                                      images_points,
                                                      images_shape,
                                                      None,
                                                      None)
    return ret, mtx, dist


def undistort_image(img, mtx, dist):
    """ Undistorts image
            mtx:
            dist:
            img:
    """
    return cv2.undistort(img, mtx, dist, None, mtx)


def undistort_images(imgs, images_object_points, images_points):
    """ Calibrates and undistorts images

    """
    # Array for Undistorted images
    undistorted_images = []
    # Calibrate camera using found corners
    image_shape = imgs[0].shape[1::-1]
    ret, mtx, dist, rvecs, tvec = cv2.calibrateCamera(object_points,
                                                      image_points,
                                                      image_shape,
                                                      None,
                                                      None)
    for img in imgs:
        # Undistort images
        undistroted_image = cv2.undistort(img, mtx, dist, None, mtx)
        undistorted_images.append(undistroted_image)

    return undistorted_images


def warp_and_transform_image(undistorted_img, src, dest):
    height, width = undistorted_img.shape[0], undistorted_img.shape[1]
    M = cv2.getPerspectiveTransform(src, dest)
    Minv = cv2.getPerspectiveTransform(dest, src)
    warped = cv2.warpPerspective(undistorted_img, M, (width, height))
    return warped, M, Minv


# Color thresholding


def other_color_thresholds(img, b_threshold=(145, 200), l_threshold=(215,255)):
    # LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    binary_b = np.zeros_like(img[:,:,0])
    B_channel = lab[:,:,2]
    binary_b[(B_channel > b_threshold[0]) & (B_channel <= b_threshold[1])] = 1
    # LUV color space 
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    L_channel = luv[:,:,0]
    binary_l = np.zeros_like(img[:,:,0])
    binary_l[(L_channel > l_threshold[0]) & (L_channel <= l_threshold[1])] = 1

    # Combined threshold
    binary_threshold = np.zeros_like(img[:,:,0])
    binary_threshold[(binary_b == 1) | (binary_l == 1)] = 1

    return binary_threshold

def red_color_threshold(img, threshold=(200, 250)):
    R_channel = img[:, :, 0]
    binary_red = np.zeros_like(R_channel)
    binary_red[(R_channel > threshold[0]) & (R_channel <= threshold[1])] = 1
    return binary_red


def hLs_color_threshold(img, threshold=(90, 255)):
    hls_image = imageutils.convert_to_hsl(img)
    S_channel = hls_image[:, :, 2]
    binary_S = np.zeros_like(S_channel)
    binary_S[(S_channel > threshold[0]) & (S_channel <= threshold[1])] = 1
    return binary_S


def combined_color_threshold(img, red_thresh, hls_thresh):
    red_binary_threshold = red_color_threshold(img, red_thresh)
    hls_binary_threshold = hLs_color_threshold(img, hls_thresh)
    other_binary_thresholds = other_color_thresholds(img)
    binary_threshold = np.zeros_like(red_binary_threshold)
    binary_threshold[(red_binary_threshold == 1) | 
                     (hls_binary_threshold == 1) | 
                     (other_binary_thresholds == 1)] = 1
    return binary_threshold

# Gradient Thresholding


def abs_sobel_thresh(gray_image, orient='x',  sobel_kernel=3, thresh=(0, 255)):
    # 1) Convert to grayscale
    # gray_image = imageutils.convert_to_gray(img)
    #
    x, y = (1, 0) if orient == 'x' else (0, 1)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(gray_image, cv2.CV_64F, x, y, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary


def combined_abs_sobelxy_thresh(gray_image,  sobel_kernel=3, thresh=(0, 255)):
    gradx_binary = abs_sobel_thresh(gray_image, 'x', sobel_kernel, thresh)
    grady_binary = abs_sobel_thresh(gray_image, 'y', sobel_kernel, thresh)
    combined = np.zeros_like(gradx_binary)
    combined[((gradx_binary == 1) & (grady_binary == 1))] = 1
    return combined


def mag_sobel_thresh(gray_image, sobel_kernel=3, mag_thresh=(0, 255)):
    # 1) Convert to grayscale
    # gray_image = imageutils.convert_to_gray(img)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))
    scaled_sobelxy = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))
    # 5) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(scaled_sobelxy)
    mag_binary[(scaled_sobelxy >= mag_thresh[0]) & (scaled_sobelxy <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return mag_binary


def dir_sobel_thresh(gray_image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # 1) Convert to grayscale
    # gray_image = imageutils.convert_to_gray(img)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(direction)
    dir_binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return dir_binary


def combined_sobel_mag_dir_thresh(gray_image, sobel_kernel=17, mag_thresh=(30, 100), dir_thresh=(0.7, 1.3)):
    mag_binary = mag_sobel_thresh(gray_image, sobel_kernel, mag_thresh)
    dir_binary = dir_sobel_thresh(gray_image, sobel_kernel, dir_thresh)
    combined = np.zeros_like(dir_binary)
    combined[((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


def combined_sobel_thresh(img,
                          abs_kernel=3,
                          mag_dir_kernel=17,
                          abs_thresh=(200, 250),
                          mag_thresh=(30, 100),
                          dir_thresh=(0.7, 1.3)):
    gray_image = imageutils.convert_to_gray(img)
    combined_sobel = np.zeros_like(gray_image)
    combined_binary_abs_sobel = combined_abs_sobelxy_thresh(gray_image, abs_kernel, abs_thresh)
    combined_binary_mag_dir_sobel = combined_sobel_mag_dir_thresh(gray_image,
                                                                  mag_dir_kernel,
                                                                  mag_thresh,
                                                                  dir_thresh)
    combined_sobel[(combined_binary_abs_sobel == 1) | (combined_binary_mag_dir_sobel == 1)] = 1
    return combined_sobel


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
# Masking


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
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


def distance_from_center(l_line, r_line, image_width):
    return None


def convolution_sliding_window(binary_warped):
    return None


def window_mask(width, height, img_ref, center, level, nonzeroy, nonzerox):
    low_y = int(img_ref.shape[0] - (level + 1) * height)
    high_y = int(img_ref.shape[0] - level * height)
    low_x = max(0, int(center - width / 2))
    high_x = min(int(center + width / 2), img_ref.shape[1])

    # Output image
    output = np.zeros_like(img_ref)
    output[low_y:high_y, low_x:high_x] = 1

    # Identify the nonzero pixels in x and y within the window
    good_inds = ((nonzeroy >= low_y) & (nonzeroy < high_y) &
                 (nonzerox >= low_x) & (nonzerox < high_x)).nonzero()[0]

#     # If you found > minpix pixels, recenter next window on their mean position
#     if len(good_inds) > minpix:
#         leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

    return output, good_inds


def histogram_sliding_window(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    height, width = binary_warped.shape[0], binary_warped.shape[1]

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)
                     ** 1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])
                       ** 2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)

    # Calculate the new radii of curvature
    left_curvature = get_rad_curv(lefty, leftx)
    right_curvature = get_rad_curv(righty, rightx)
    print('Left curvature: {}m, Right curvature: {}m'.format(left_curverad, right_curverad))

    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, width)
    plt.ylim(height, 0)
    plt.show()


def find_window_centroids(image, window_width, window_height, margin, previous_l_center=None, previous_r_center=None):

    # Identify image height and width
    height, width = image.shape[0], image.shape[1]
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template
    if not previous_l_center and not previous_r_center:
        # print('Yay')
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(3 * height / 4):, :int(width / 2)],
                       axis=0)  # *3 to get the bottom quarter
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        r_sum = np.sum(image[int(3 * height / 4):, int(width / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum))-window_width/2+int(image.shape[1]/2)
        starting_level = 1
    else:
        print('Nay')
        # Add what we found for the first layer
        window_centroids.append((previous_l_center, previous_r_center))
        l_center = previous_l_center
        r_center = previous_r_center
        starting_level = 0

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each other (save the last one) layer looking for max pixel locations
    for level in range(starting_level, (int)(height / window_height)):
        # convolve the window into the vertical slice of the image
        # 720 - 160 second window : 720 - 80
        image_layer = np.sum(image[int(height - (level + 1) * window_height)                                   : int(height - level * window_height), :], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as an offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, width))\
            #         x = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
#         if np.abs(x - l_center)
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

#         print(conv_signal[int(x)])
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, width))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset

        # Add what we found for that layer
        window_centroids.append((l_center, r_center))
    return window_centroids


def draw_lines_windows(warped, window_width, window_height, nonzerox, nonzeroy, window_centroids):
    left_indices, right_indices = [], []
    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # Go through each level and draw the windows
    for level in range(0, len(window_centroids)):
        # Window_mask is a function to draw window areas
        l_mask, l_inds = window_mask(window_width,
                                     window_height,
                                     warped,
                                     window_centroids[level][0],
                                     level,
                                     nonzeroy,
                                     nonzerox)
        r_mask, r_inds = window_mask(window_width,
                                     window_height,
                                     warped,
                                     window_centroids[level][1],
                                     level,
                                     nonzeroy,
                                     nonzerox)

        left_indices.append(l_inds)
        right_indices.append(r_inds)
        # Add graphic points from window mask here to total pixels found
        l_points[(l_points == 255) | ((l_mask == 1))] = 255
        r_points[(r_points == 255) | ((r_mask == 1))] = 255

    # Draw the results
    # add both left and right window pixels together
    template = np.array(r_points + l_points, np.uint8)
    zero_channel = np.zeros_like(template)  # create a zero color channel
    template = np.array(cv2.merge((zero_channel, template, zero_channel)),
                        np.uint8)  # make window pixels green
    # making the original road pixels 3 color channels
    warpage = np.dstack((warped, warped, warped))*255
    # overlay the orignal road image with window results
    output = cv2.addWeighted(warpage, 1, template, 0.8, 0.0)
    return output, np.concatenate(left_indices), np.concatenate(right_indices)


def polyfit_lines(leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each
    # Quadratic cofficent A
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit


def compute_polyfit(left_fit, right_fit, out_img):
    # Generate x and y values for plotting
    height, width, _ = out_img.shape
    ploty = np.linspace(0, height - 1, num=height)
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, width)
    # plt.ylim(height, 0)
    return ploty, left_fitx, right_fitx


def get_rad_curv(y_vals, x_vals, ym_per_pix=30/720, xm_per_pix=3.7/700):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/700  # meters per pixel in x dimension
    fit_cr = np.polyfit(y_vals * ym_per_pix, x_vals * xm_per_pix, 2)
    y_eval = np.max(y_vals)
    return ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2 * fit_cr[0])


def get_curvature(lefty, leftx, righty, rightx):
    # Calculate the new radii of curvature
    left_curverad = get_rad_curv(lefty, leftx)
    right_curverad = get_rad_curv(righty, rightx)
    # print('Left curvature: {}, Right curvature: {}\n'.format(left_curverad, right_curverad))
    # print("Difference between both line's curvatures {} ".format(np.abs(left_curverad - right_curverad)))
    # print("ok {}, {}".format(left_curverad, right_curverad))
    return np.average([left_curverad, right_curverad])


def get_vehicle_position(img, left_fitx, right_fitx, xm_per_pix=3.7/700):
    height, width, _ = img.shape
    car_center_bottom = width / 2 # becausze the car is the center  of the image
    lane_center = (left_fitx[height - 1] + right_fitx[height - 1]) / 2 
    # print((car_center_bottom - lane_center) * xm_per_pix)
    # exit
    return (car_center_bottom - lane_center) * xm_per_pix



def draw_drivable_area(warped, undist_image, ploty, left_fitx, right_fitx, Minv):
    # ploty = np.linspace(0, height - 1, num=height)
    # Create an image to draw the lines on
    new_copy = np.copy(undist_image)
    if left_fitx is None or right_fitx is None:
        return undist_image
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.polylines(color_warp, np.int32([pts_left]),
                  isClosed=False, color=(255, 0, 255), thickness=20)
    cv2.polylines(color_warp, np.int32([pts_right]),
                  isClosed=False, color=(0, 255, 255), thickness=20)
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist_image.shape[1], undist_image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(new_copy, 1, newwarp, 0.7, 0)
    return result

from collections import deque


class LaneLineFinder:

    SAMPLE_FRAMES = 15

    def __init__(self, mtx, dist, keras_model=None):
        self.previous_frames = deque(maxlen= self.SAMPLE_FRAMES)
        self.left_lane_fits = deque(maxlen= self.SAMPLE_FRAMES)
        self.right_lane_fits = deque(maxlen= self.SAMPLE_FRAMES)
        self.curvatures = deque(maxlen= self.SAMPLE_FRAMES)
        self.center_values = deque(maxlen= self.SAMPLE_FRAMES)
        self.mtx = mtx
        self.dist = dist
        self.keras_model = keras_model
        self.previous_image_centroids = None
        self.drivable = deque(maxlen=self.SAMPLE_FRAMES)

    def average_frame_sampling(self, frame, previous_frames):
        previous_frames.append(frame)
        if len(previous_frames) > 0:
            frame = np.mean(previous_frames, axis=0, dtype=np.int32) 
            # line = tuple(map(tuple, line))

        return frame

    def process_image(self, image):
        height, width, _ = image.shape

        # Color Thresholds
        red_thresh = (220, 250)
        hls_thresh = (90, 255)
        hls2_thresh = (170, 255)
        # Gradient Thresholds
        xy_threshold = (20, 100)
        mag_threshold = (70, 100)
        dir_threshold = (1.1, 1.3)

        image_offset = 10
        src = np.float32([[width * 0.45, height * 0.63]  # Top left vertix 60% if the image's hight
                          , [width * 0.10, height * 0.95]  # Bottom left
                          , [width * 0.94, height * 0.95]  # Bottom right
                          , [width * 0.56, height * 0.63]])  # Top right vetrix

        dest = np.float32([[image_offset, 0],  # Top left
                           [image_offset, height],  # Bottom left
                           [width - image_offset, height],  # Bottom right
                           [width - image_offset, 0]])  # Top right

        # Undistort image
        undistorted_image = undistort_image(image, self.mtx, self.dist)

        if not self.keras_model:
            # Thresholding
            # Color Thresholding
            color_binary_threshold = combined_color_threshold(
                undistorted_image, red_thresh, hls2_thresh)
            # Gradient Thresholding (Sobel)
            sobel_binary_threshold = combined_sobel_thresh(undistorted_image,
                                                           abs_kernel=3,
                                                           mag_dir_kernel=17,
                                                           abs_thresh=xy_threshold,
                                                           mag_thresh=mag_threshold,
                                                           dir_thresh=dir_threshold)
            combined_color_gradient = np.zeros_like(color_binary_threshold)
            combined_color_gradient[(color_binary_threshold == 1) |
                                    (sobel_binary_threshold == 1)] = 1
            image_to_warp = combined_color_gradient
            # plt.imshow(image_to_warp)
            # plt.show()
            # print(image_to_warp.dtype)
            # print(image_to_warp.shape)
            # Perspective transform (Warp)
            warped_image, M, Minv = warp_and_transform_image(image_to_warp, src, dest)
            # Smooth the image (Gaussian blur)
            warped_image = gaussian_blur(warped_image, 11)
            # plt.imshow(warped_image, cmap="gray")
            # plt.show()
            # Detecting lane lines
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = warped_image.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # window settings
            window_width = 100
            window_height = height / 6  # Break image into 5 vertical layers since image height is 720
            margin = 100  # How much to slide left and right for searching

            # Calculate lanes centroids
            # Get lanes centroids (Windows)
            window_centroids = find_window_centroids(warped_image, window_width, window_height, margin)

            # Draw detected windows
            output, left_indices, right_indices = draw_lines_windows(warped_image,
                                                                     window_width,
                                                                     window_height,
                                                                     nonzerox,
                                                                     nonzeroy,
                                                                     window_centroids)
            # Extract left and right line pixel positions
            leftx = nonzerox[left_indices]
            lefty = nonzeroy[left_indices]
            rightx = nonzerox[right_indices]
            righty = nonzeroy[right_indices]
            # Get poly fits for the lane lines
            if len(leftx) > 0 and len(lefty) > 0 and len(rightx) > 0 and len(righty) > 0:
                # Get the polyfits for the lane lines
                left_fit, right_fit = polyfit_lines(leftx, lefty, rightx, righty)
                
                # Color lane lines
                # making the original road pixels 3 color channels
                out_img = np.dstack((warped_image, warped_image, warped_image)) * 255
                out_img[nonzeroy[left_indices], nonzerox[left_indices]] = [255, 0, 0]
                out_img[nonzeroy[right_indices], nonzerox[right_indices]] = [0, 0, 255]
                # Draw polyfit and lane lines
                ploty, left_fitx, right_fitx = compute_polyfit(left_fit,
                                                                       right_fit,
                                                                       out_img)
                # Get curvature
                lane_curvature = self.average_curvature(get_curvature(lefty, leftx, righty, rightx))
                # if len(left_fitx) > 0 and len(right_fitx) > 0
                # Get car position
                car_position = get_vehicle_position(out_img, left_fitx, right_fitx)

                actual_size=3.7/700
                # print(np.abs(leftx[0] - rightx[0]) * actual_size) 
                lane_dist_high = np.abs(leftx[0] - rightx[0]) 
                lane_dist_low = np.abs(leftx[-1] - rightx[-1])

                print(lane_dist_high)
                print(lane_dist_low)

                left_curverad = get_rad_curv(lefty, leftx)
                right_curverad = get_rad_curv(righty, rightx)
                print('left curv', left_curverad)
                print('right curv', right_curverad)

                if (lane_dist_low < 750) or (lane_dist_high < 750):
                    # print(lane_dist_high)
                    # print(lane_dist_low)
                    # print('LOW')
                    diff = left_curverad - right_curverad
                    if np.abs(diff) >= 200:
                        if diff < 0 and right_curverad > 380:
                            right_fitx = right_fitx 
                            left_fitx = right_fitx - 900
                        elif diff > 0 and left_curverad > 380: 
                            right_fitx = left_fitx + 900
                            left_fitx = left_fitx
                    else:
                        right_fitx, left_fitx = np.array([]), np.array([])

                right_fitx = self.average_lane_sampling(right_fitx, self.right_lane_fits)
                left_fitx = self.average_lane_sampling(left_fitx, self.left_lane_fits)
                masked_lane_image = draw_drivable_area(
                    warped_image, image, ploty, left_fitx, right_fitx, Minv)
                write_text_on_image(masked_lane_image, int(lane_curvature), car_position)
                return masked_lane_image
            else:
                return image
        else:
            perdicted_image = self.keras_model.predict(undistorted_image).astype('uint8')
            height, width, _ = undistorted_image.shape
            # Get lane polyFit
            nonzero = perdicted_image[:,:,2].nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            curvature = self.average_curvature(get_rad_curv(nonzeroy, nonzerox) if len(nonzeroy) > 0 and len(nonzerox) > 0 else 0)
            new_copy = np.copy(undistorted_image)
            frame_to_return = perdicted_image#self.average_frame_sampling(image_to_warp, self.previous_frames)
            write_text_on_image(new_copy, int(curvature))
            return cv2.addWeighted(new_copy, 1, frame_to_return.astype('uint8')*255, 0.7, 0)
    
    def average_lane_sampling(self, line_fit, previous_fits):
        if line_fit.size > 0:
            # print('append')
            previous_fits.append(line_fit)
        
        if len(previous_fits) > 0:
            line_fit = np.mean(previous_fits, axis = 0, dtype=np.int32)
            
        return line_fit

    def average_curvature(self, curvature):
        if curvature > 0:
            self.curvatures.append(curvature)
        if len(self.curvatures) > 0:
            curvature = np.mean(self.curvatures, axis = 0, dtype=np.int32)
        return curvature
    def average_car_position(self, center):
        self.center_values.append(center)
        if len(self.center_values) > 0:
            center = np.mean(self.center_values, axis = 0, dtype=np.int32)
        return center
def write_text_on_image(img, curv, center):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (int(img.shape[1] * 0.32), int(img.shape[0] * 0.10))
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    # img[:100, :] -= 5 
    bottomLeftCornerOfText2 = (int(img.shape[1] * 0.32), int(img.shape[0] * 0.20))
    curv = 'Straigh' if curv > 3000 else curv
    cv2.putText(img, 'Curvature is ({}) meters'.format(curv), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    cv2.putText(img, 'The Car is ({}) meters of center'.format(round(center, 1)), 
        bottomLeftCornerOfText2, 
        font, 
        fontScale,
        fontColor,
        lineType)