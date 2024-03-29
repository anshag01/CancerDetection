import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    # Convert to grayscale (if required)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize the image to (224, 224)
    resized_img = cv2.resize(gray_img, (224, 224))
    # Normalize pixel values to the range 0-1
    normalized_img = resized_img / 255.0
    # Expand dimensions to match the input shape for the CNN (1, 224, 224, 1)
    final_img = np.expand_dims(normalized_img, axis=[0, -1])
    return final_img
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Using sigmoid for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.fit(train_images, train_labels, epochs=10, validation_data=(validation_images, validation_labels))

processed_img = preprocess_image('path_to_new_image.jpg')
prediction = model.predict(processed_img)
print("Cancer Detected" if prediction[0][0] > 0.5 else "No Cancer Detected")
