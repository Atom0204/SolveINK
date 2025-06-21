# custom_dataset.py
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class CustomMathDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, label_column='label'):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_column = label_column

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image_path'])
        image = Image.open(img_name).convert('L')  # Convert to grayscale
        label = str(self.data_frame.iloc[idx][self.label_column])

        if self.transform:
            image = self.transform(image)

        return image, label