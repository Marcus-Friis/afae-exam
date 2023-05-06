import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Classifier(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()

        self.sequence = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(1, -1),
            nn.Linear(128*4*4, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.sequence(x)
    

if __name__ == '__main__':
    x = torch.randn(64, 3, 64, 64)
    model = Classifier(8)
    print(model(x).shape)