import cv2
import numpy as np


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (600, 450))
    return resized_img


def load_preprocess_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_contours(thresh):
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours


def classify_shapes(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

    # print("Circularity:", circularity, "Vertices:", len(approx))
    if len(approx) > 4 and circularity > 0.6:
        return 'circular'
    elif len(approx) < 5 and circularity < 0.7:
        return 'pointed'
    else:
        return 'irregular'

