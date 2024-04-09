import cv2
import numpy as np


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (600, 450))
    return resized_img


# def classify_shape(contour, thresh):
#     (x, y, w, h) = cv2.boundingRect(contour)
#     aspect_ratio = float(w) / h
#     print("Aspect Ratio:", aspect_ratio)
#     circularity = (4 * np.pi * cv2.contourArea(contour)) / (cv2.arcLength(contour, True) ** 2)
#     print("Circularity:", circularity)
#
#     # Re-integrate aspect ratio into the decision for spherical-like shapes
#     if 0.8 <= circularity <= 1.2:
#         if 0.9 <= aspect_ratio <= 1.1:
#             return "sphere"
#         else:
#             return "oval"  # Oval-like but still circular
#     elif 1.2 < circularity <= 1.8:
#         return "oval"
#     else:
#         return "pointed"

# def detect_dark_patch_shape(image_path):
#     # Read the image
#     image = cv2.imread(image_path)
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Apply thresholding to get a binary image
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     # print(thresh)
#
#     # Find contours
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Sort the contours by area and then take the largest one
#     sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     if not sorted_contours:
#         return "No contours found", None
#
#     # We assume the largest contour is the dark patch
#     c = sorted_contours[0]
#
#     # Classify the shape
#     shape = classify_shape(c, thresh)
#
#     return shape

# def detect_dark_patch_shape2(image_path):
#     # Read the image
#     image = cv2.imread(image_path)
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Apply thresholding to get a binary image
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     # Find contours
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Sort the contours by area and then take the largest one
#     sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     if not sorted_contours:
#         return "No contours found", None
#
#     # We assume the largest contour is the dark patch
#     c = sorted_contours[0]
#
#     # Calculate the perimeter of the contour
#     peri = cv2.arcLength(c, True)
#
#     # Use a lower approximation tolerance to get a more precise shape
#     approx = cv2.approxPolyDP(c, 0.01 * peri, True)
#
#     # The number of vertices of the contour will tell us the shape
#     vertices = len(approx)
#     shape_description = f"polygon with {vertices} vertices"
#     print(shape_description)
#
#     # Draw the contour on the image
#     cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
#
#     # Save the image with shape detected for visualization
#     output_path = '/mnt/data/dark_patch_shape_detected.png'
#     cv2.imwrite(output_path, image)
#
#     return output_path, vertices

def load_preprocess_image(file_path):
    # Load the image in grayscale
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Apply a Gaussian blur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Threshold the image - Otsu's method calculates an optimal threshold value
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


def extract_contours(thresh):
    # Find contours using the thresholded image
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    return contours


def classify_shapes(contours):
    shapes = []
    for contour in contours:
        # Calculate contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2))

        # Simplify the contour to remove minor variations
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        # Determine shape category based on the properties
        if len(approx) > 8 and circularity > 0.7:
            shapes.append('circular')  # More than 8 vertices and circular shape
        elif len(approx) < 5:
            shapes.append('angular')  # Fewer vertices, more angular
        else:
            shapes.append('irregular')  # Does not fit into other categories

    return shapes

