import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.fairface import FairfaceData
from src.models import Classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 8)
model = model.to(device)
transform = weights.transforms()

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((64, 64), antialias=True)
#     ])

traindata = FairfaceData('fairface', 'fairface_label_train.csv', transform=transform)
trainloader = DataLoader(traindata, batch_size=64, shuffle=True)

valdata = FairfaceData('fairface', 'fairface_label_val.csv', transform=transform)
valloader = DataLoader(valdata, batch_size=64, shuffle=True)


# model = Classifier(8).to(device)
    
n_epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# calculate accuracy
def accuracy(y_hat, y):
    y_hat = torch.argmax(y_hat, dim=1)
    y = torch.argmax(y, dim=1)
    return (y_hat == y).float().mean()

for epoch in range(n_epochs):
    model.train()
    for images, y, race, gender in trainloader:
        images = images.to(device)
        y = y.to(device)
        # plt.imshow(  images[0].permute(1, 2, 0))
        # plt.show()
        y_hat = model(images)
        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break

    model.eval()
    val_losses = []
    val_accs = []
    for images, y, race, gender in valloader:
        images = images.to(device)
        y = y.to(device)
        
        y_hat = model(images)
        loss = criterion(y_hat, y)
        val_losses.append(loss.item())
        val_accs.append(accuracy(y_hat, y).item())
    print('EPOCH', epoch, 'Validation loss: ', np.mean(val_losses), 'Validation accuracy: ', np.mean(val_accs))
