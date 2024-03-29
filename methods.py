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


# def build_model():
#     model = Sequential([
#         Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
#         MaxPooling2D(2, 2),
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D(2, 2),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dense(1, activation='sigmoid')  # Using sigmoid for binary classification
#     ])
#
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model
#
# model = build_model()
# model.fit(train_images, train_labels, epochs=10, validation_data=(validation_images, validation_labels))
#
# processed_img = preprocess_image('path_to_new_image.jpg')
# prediction = model.predict(processed_img)
# print("Cancer Detected" if prediction[0][0] > 0.5 else "No Cancer Detected")
