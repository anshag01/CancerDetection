import os

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
