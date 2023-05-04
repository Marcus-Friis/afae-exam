import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from PIL import Image
from tqdm import tqdm

class FairfaceData(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.annotations = pd.read_csv(os.path.join(self.root_dir, 'fairface_label_train.csv'))
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        
        class_map = {'Latino_Hispanic': 0,
             'East Asian': 1,
             'Indian': 2,
             'Middle Eastern': 3,
             'Black': 4,
             'Southeast Asian': 5,
             'White': 6
             }
        
        label = self.annotations.iloc[index, 3]
        label = class_map[label]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor()
        ])
    
    data = FairfaceData('fairface', transform=transform)

