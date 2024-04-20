import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

import methods


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


def list_image_files(data_folder_path: str) -> list:
    """List the base filenames of JPG images in the specified directory."""
    files = [f for f in os.listdir(data_folder_path) if f.endswith(".jpg")]
    return [os.path.splitext(f)[0] for f in files]


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
    data_folder_path: str,
    features_path: str,
    labels_df: pd.DataFrame,
    export: bool = False,
) -> pd.DataFrame:
    """Merge image features with labels into a single DataFrame."""
    filenames = list_image_files(data_folder_path)
    temp_files = extract_image_ids(filenames)

    label_data = temp_files.merge(labels_df, on="image_id", how="left")
    features = load_features(features_path)

    merged_data = features.merge(label_data, left_index=True, right_on="image_id")

    if export:
        export_path = os.path.join(
            os.path.dirname(features_path), "pandas_features.csv"
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
