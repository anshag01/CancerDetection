import math

import cv2
import numpy as np


def rotatedRectWithMaxArea(w, h, angle):
    """
    adapted from: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


def rotate_image(image, angle):
    """
    :param image:
    :param angle: angle in degree
    :return: rotated image without boundary box
    """
    angle_rad = math.radians(angle)

    # image dimensions
    image_height, image_width = image.shape[0:2]

    # get the width and height of the largest rectangle with the same aspect ratio
    wr, hr = rotatedRectWithMaxArea(image_width, image_height, angle_rad)

    # compute the rotation matrix for the given angle
    rotation_matrix = cv2.getRotationMatrix2D(
        (image_width / 2, image_height / 2), angle, 1.0
    )

    # rotate image
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (image_width, image_height), flags=cv2.INTER_LINEAR
    )

    # compute the size of the cropped image
    crop_width = int(wr)
    crop_height = int(hr)

    # ensure the crop width and height are not larger than the image dimensions
    crop_width = min(crop_width, image_width)
    crop_height = min(crop_height, image_height)

    # calculate the top-left corner of the cropped area
    top_left_x = max(0, int((image_width - crop_width) / 2))
    top_left_y = max(0, int((image_height - crop_height) / 2))

    # crop the image to the calculated dimensions
    cropped_image = rotated_image[
        top_left_y : top_left_y + crop_height, top_left_x : top_left_x + crop_width
    ]

    # Save or show the cropped image
    return cropped_image


def crop_image_to_size(image, target_width, target_height):
    """
    :param image:
    :param target_width:
    :param target_height:
    :return: cropped image to a specific target size
    """
    # Get dimensions of the original image
    h, w = image.shape[:2]

    # Calculate coordinates to crop the image to the new size
    start_x = w // 2 - target_width // 2
    start_y = h // 2 - target_height // 2
    end_x = start_x + target_width
    end_y = start_y + target_height

    # Crop the image using array slicing
    cropped_image = image[start_y:end_y, start_x:end_x]
    return cropped_image


def oversample_image(image, target_width, target_height):
    h, w = image.shape[:2]

    # Define the scale factor
    scale_factor = np.max([target_width / w, target_height / h])

    # Calculate the new dimensions
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    new_dimensions = (new_width, new_height)

    # Resize the image
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_CUBIC)

    return resized_image


def crop_rotate(image, angle):
    """
    :param image:
    :param angle: angle in degree
    :return: rotated and axis-aligned cropped imagen with original dimensions
    """
    image_rotated = rotate_image(image, angle)

    h, w = image.shape[:2]

    oversample = oversample_image(image_rotated, w, h)

    return crop_image_to_size(oversample, w, h)
