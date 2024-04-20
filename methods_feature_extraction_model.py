import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

import methods

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


from sklearn import datasets
from sklearn.manifold import TSNE

class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith((".jpg", ".png"))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Do preprocessing here
        image_path = self.images[idx]
        rgb_image_arr = methods.convert_rgb(image_path)
        normalised_img = methods.z_normalization(rgb_image_arr)
        image = Image.fromarray(normalised_img.astype("uint8"), "RGB")
        image_tensor = self.transform(image) if self.transform else image
        key = os.path.basename(image_path).removesuffix(".jpg").removesuffix(".png")
        return key, image_tensor

def extract_image_ids(filenames: list) -> pd.DataFrame:
    """Extract image IDs from filenames assuming format includes ID as the last two underscore-separated parts."""
    image_ids = ["_".join(name.split("_")[-2:]) for name in filenames]
    return pd.DataFrame({"filename": filenames, "image_id": image_ids})


def load_features(features_path: str) -> pd.DataFrame:
    """Load features from a JSON file and transpose the dataframe."""
    if not features_path.endswith(".json"):
        raise ValueError("Invalid file type: JSON expected")
    features = pd.read_json(features_path).T
    return features


def merge_features_with_labels(
    features_path: str,
    labels_df: pd.DataFrame,
    export: bool = False,
) -> pd.DataFrame:
    """Merge image features with labels into a single DataFrame."""
    features = load_features(features_path)

    filenames = features.T.columns.to_numpy()
    temp_files = extract_image_ids(filenames)

    label_data = temp_files.merge(labels_df, on="image_id", how="left")


    merged_data = features.merge(label_data, left_index=True, right_on="image_id")

    if export:
        export_path = os.path.join(
            os.path.dirname(features_path), features_path.split(".")[-2].split("/")[-1] +  "_pandas.csv"
        )
        merged_data.to_csv(export_path)

    return merged_data


def not_oversampled_images(features_dataframe: pd.DataFrame) -> list[bool]:
    """
    Generate a list indicating whether each image in the dataframe should be included
    in testing based on whether it has been augmented.

    :param features_dataframe: A DataFrame containing image metadata with columns 'filename' and 'image_id'.
    :return: A list of booleans where True indicates the image has not been augmented,
                and False indicates it has.
    """

    # Extract augmented image IDs
    augmented_ids = set(
        "_".join(
            filename.split("_")[-2:]
        )  # Assuming the ID is in the last two parts of the filename
        for filename in features_dataframe.filename
        if "augmented" in filename
    )

    # Determine inclusion in testing for each image
    include_in_testing = [
        image_id not in augmented_ids for image_id in features_dataframe.image_id
    ]

    return include_in_testing


def calculate_test_size(dataframe, test_size, include_in_testing):
    """
    Calculate the adjusted test size for splitting the dataset, excluding oversampled entries.

    :param dataframe: The complete dataframe.
    :param test_size: The desired proportion of the test set size relative to the unique images.
    :param include_in_testing: Boolean array indicating which images are not oversampled.
    :return: Adjusted test size.
    """
    unique_image_count = len(np.unique(dataframe.image_id))
    valid_image_count = np.sum(include_in_testing)
    return test_size * unique_image_count / valid_image_count



def calculate_metrics(y_test, y_pred):
    """
    Calculates and prints the accuracy, precision, recall, and F1 score for the given test labels and predictions.

    :param y_test: True labels for the test set.
    :param y_pred: Predicted labels for the test set.
    :return:
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")




def plot_confusion_matrix(y_test, y_pred, print_metrics=True):
    """
    Plots the confusion matrix for the given test labels and predictions.

    :param y_test: True labels for the test set.
    :param y_pred: Predicted labels for the test set.
    :return:
    """
    if print_metrics:
        calculate_metrics(y_test, y_pred)

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Initialize the ConfusionMatrixDisplay object with the confusion matrix
    cmd = ConfusionMatrixDisplay(conf_matrix)

    # Plot the confusion matrix
    cmd.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def plot_low_dim_components(x_train_low_dim, y_train, label="PCA", component_1=0, component_2=1):
    # Scatter plot of the first two PCA components
    # Here, X_pca[:, 0] is the first component, X_pca[:, 1] is the second component
    plt.figure(figsize=(10, 7))
    plt.scatter(
        x_train_low_dim[y_train == 0, component_1],
        x_train_low_dim[y_train == 0, component_2],
        c="blue",
        label="Non-Cancerous",
        alpha=0.5,
    )  # Non-cancerous in blue
    plt.scatter(
        x_train_low_dim[y_train == 1, component_1],
        x_train_low_dim[y_train == 1, component_2],
        c="red",
        label="Cancerous",
        alpha=0.5,
    )  # Cancerous labeled in red

    # Adding labels and title
    plt.xlabel(f"{label} Component {component_1}")
    plt.ylabel(f"{label} Component {component_2}")
    plt.title(f"{label} of Image Data")
    plt.legend()