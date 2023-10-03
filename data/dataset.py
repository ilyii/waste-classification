import os

import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class WasteDataset(Dataset):
    """Waste dataset.
    Folder structure:
    data
    ├── class1
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...

    Args:
        data_path (string): Path to the data folder.
        classes (list): List of classes (folder names) to use.
        transform (callable, optional): Optional transform to be applied.

    """

    def __init__(self, data_path, classes, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.classes = classes
        self.image_info = []

        for folder in tqdm(os.listdir(data_path), unit="class folders"):
            if folder in self.classes:
                folder_path = os.path.join(data_path, folder)
                for image_filename in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_filename)
                    label = folder.strip()
                    width, height = Image.open(image_path).size
                    self.image_info.append(
                        (image_path, self.classes.index(label), width, height)
                    )

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        image_path, encoded_label, width, height = self.image_info[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, encoded_label

    def get_classes(self):
        return self.classes
