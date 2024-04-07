import cv2
import numpy as np


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray_img.dtype)
    # Resize the image to (600, 450)
    resized_img = cv2.resize(gray_img, (600, 450))
    return resized_img

