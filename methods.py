import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def convert_grayscale(image_path):
    # Load the image
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #resized_img = cv2.resize(gray_img, (600, 450))
    return gray_img

def convert_rgb(image_path):
    """
    Converts the image to normalised RGB format
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    normalised_img = img / 255.0
    return normalised_img

def color_distribution(img):
    """
    Returns the color distribution of the image
    """
    df = pd.DataFrame(img.reshape(-1, 3), columns=['R', 'G', 'B'])
    color_distribution = df.describe()
    return color_distribution

def create_rgb_histogram(img):
    """
    Create one RGB histogram based on the distribution of the image
    """
    img = cv2.imread(img)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()
    return histr
    
def get_contrast(img):
    """
    Returns the contrast of the image
    """
    img = convert_grayscale(img)
    std_dev = np.std(img) #Contrast
    return std_dev

