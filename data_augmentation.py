import math

import cv2
import numpy as np
import random
import os
import methods
import shutil
from sklearn.model_selection import train_test_split


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
    Crop an image to a specific target size centered on the original image.
    :param image: Input image as a NumPy array.
    :param target_width: Target width for the cropped image.
    :param target_height: Target height for the cropped image.
    :return: Cropped image as a NumPy array.
    """
    # Get dimensions of the original image
    h, w = image.shape[:2]

    # Check if the target size is larger than the original size
    if target_width > w or target_height > h:
        raise ValueError("Target size must be smaller than the original size.")

    # Calculate coordinates to crop the image to the new size
    start_x = max(0, w // 2 - target_width // 2)
    start_y = max(0, h // 2 - target_height // 2)
    end_x = min(w, start_x + target_width)
    end_y = min(h, start_y + target_height)

    # Ensure the coordinates do not exceed the image boundaries
    start_x = min(start_x, w - target_width)
    start_y = min(start_y, h - target_height)

    # Crop the image using array slicing
    cropped_image = image[start_y:end_y, start_x:end_x]
    return cropped_image


def oversample_image(image, target_width, target_height):
    """
    Resize an image to cover the specified dimensions by oversampling if necessary.
    The image will be enlarged to meet or exceed the target dimensions while maintaining aspect ratio.

    :param image: Input image as a NumPy array.
    :param target_width: The target width.
    :param target_height: The target height.
    :return: Resized image that covers the specified dimensions.
    """
    # Original dimensions
    h, w = image.shape[:2]

    # Determine the scale factor needed to meet or exceed target dimensions while maintaining aspect ratio
    scale_factor = max(target_width / w, target_height / h)

    # Calculate the new dimensions that meet or exceed the target size
    new_width = int(np.ceil(w * scale_factor))
    new_height = int(np.ceil(h * scale_factor))
    new_dimensions = (new_width, new_height)

    # Resize the image using cubic interpolation
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


def create_directories(path_list):
    """Create directories for each path in the list."""
    for path in path_list:
        os.makedirs(path, exist_ok=True)

def filter_files_in_directory(directory, extension=".jpg"):
    """List files in directory with specific extension and return without extension."""
    return [f.split(".")[0] for f in os.listdir(directory) if f.endswith(extension) and os.path.isfile(os.path.join(directory, f))]

def copy_files(files, destination):
    """Copy files to the specified destination."""
    for file in files:
        shutil.copy(file, destination)

def split_data_with_labels(dataset_directory, processed_folder_path, labels_df, train_size=0.8):
    train_dir = os.path.join(processed_folder_path, 'train')
    test_dir = os.path.join(processed_folder_path, 'test')
    create_directories([train_dir, test_dir])

    files = filter_files_in_directory(dataset_directory)
    labels_df = labels_df[labels_df['image_id'].isin(files)]

    for class_label in labels_df['cancer'].unique():
        class_df = labels_df[labels_df['cancer'] == class_label]
        images = (class_df['image_id'] + ".jpg").tolist()
        images_full_path = [os.path.join(dataset_directory, img) for img in images]
        train_imgs, test_imgs = train_test_split(images_full_path, train_size=train_size, random_state=42)

        class_train_dir = os.path.join(train_dir, str(class_label))
        class_test_dir = os.path.join(test_dir, str(class_label))
        create_directories([class_train_dir, class_test_dir])

        copy_files(train_imgs, class_train_dir)
        copy_files(test_imgs, class_test_dir)

def oversample_train_data(train_directory):
    random.seed(42)
    elements_per_class = []
    class_labels = []

    for class_folder in os.listdir(train_directory):
        class_path = os.path.join(train_directory, class_folder)
        if os.path.isdir(class_path):
            files = filter_files_in_directory(class_path)
            class_labels.append(class_folder)
            elements_per_class.append(len(files))

    max_images = max(elements_per_class)
    augment_images_in_classes(train_directory, class_labels, max_images)

def augment_images_in_classes(train_directory, class_labels, max_images):
    """Oversample training data by augmenting images within each class."""
    for class_folder in class_labels:
        class_path = os.path.join(train_directory, class_folder)
        images = os.listdir(class_path)
        original_images = [img for img in images if img.endswith('.jpg') and 'augmented' not in img]

        while len(images) < max_images:
            augment_and_save_image(class_path, original_images, images)

def augment_and_save_image(class_path, original_images, images):
    """Augment an image with random rotation and save it."""
    image_to_augument = random.choice(original_images)
    image_path = os.path.join(class_path, image_to_augument)
    angle = random_rotation_angle()
    new_image = rotate_image(cv2.imread(image_path), angle)  # Assuming a rotate_image function exists
    new_image_name = f"augmented_{angle:.0f}deg_{image_to_augument}"
    new_image_path = os.path.join(class_path, new_image_name)
    cv2.imwrite(new_image_path, new_image)
    images.append(new_image_name)

def random_rotation_angle():
    """Generate a random rotation angle within predefined intervals."""
    diff_angle = 15  # degree
    intervals = [
        (0, 0 + diff_angle),
        (90 - diff_angle, 90 + diff_angle),
        (180 - diff_angle, 180 + diff_angle),
        (270 - diff_angle, 270 + diff_angle),
        (360 - diff_angle, 360)
    ]
    return random.uniform(*random.choice(intervals))
