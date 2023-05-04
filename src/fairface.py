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
    def __init__(self, root_dir, csv, target='age', transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.annotations = pd.read_csv(os.path.join(self.root_dir, csv))
        self.transform = transform
        self.age = pd.get_dummies(self.annotations['age'], drop_first=True).astype(float)
        self.gender = pd.get_dummies(self.annotations['gender'], drop_first=True).astype(float)
        self.race = pd.get_dummies(self.annotations['race'], drop_first=True).astype(float)
        self.target = target

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        
        age = torch.from_numpy(self.age.iloc[index].to_numpy())
        race = torch.from_numpy(self.race.iloc[index].to_numpy())
        gender = torch.from_numpy(self.gender.iloc[index].to_numpy())
        
        if self.transform:
            image = self.transform(image)
        
        return image, age, race, gender


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor()
        ])
    
    data = FairfaceData('fairface', transform=transform)

