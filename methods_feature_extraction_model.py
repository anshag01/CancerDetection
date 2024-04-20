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


def merge_features_with_labels(
    data_folder_path: str, features_path: str, labels_df: pd.DataFrame, export=False
):
    # get all files in folder
    files = os.listdir(data_folder_path)
    files = [f.split(".")[-2] for f in files if f.endswith(".jpg")]

    temp_files = pd.DataFrame(files, columns=["filename"])
    temp_files["image_id"] = temp_files.filename.apply(
        lambda x: x.split("_")[-2] + "_" + x.split("_")[-1]
    )

    label_ = temp_files.merge(labels_df, on="image_id", how="left")

    if features_path.endswith(".json"):
        features = pd.read_json(features_path)
        features = features.T
    else:
        assert False, "Invalid File Type"

    merged_data = features.merge(label_, left_index=True, right_on="image_id")

    if export:
        # export cnn features
        merged_data.to_csv(
            os.path.join(os.path.dirname(features_path), "pandas_features.csv")
        )

    return merged_data
