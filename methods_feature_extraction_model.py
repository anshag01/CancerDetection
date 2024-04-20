import os

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
