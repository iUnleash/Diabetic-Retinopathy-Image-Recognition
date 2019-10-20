# -*- coding: utf-8 -*-

## Starter Code for the Final Project

import pandas as pd
import numpy as np
from PIL import Image
import torch
from skimage import io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import math


class APTOSDataset(Dataset):
    """Eye images dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        self.eye_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.eye_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir,
                                self.eye_frame.iloc[idx,0] + '.png')
        
        image = Image.open(img_name)
        image = self.transform(image)

        return (image,self.eye_frame.diagnosis[idx])
    
    
train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        ])
    
    
eye_dataset = APTOSDataset(csv_file='train.csv',
                           root_dir='train_images',
                           transform=train_transform)


batch_size = 1000
n_epochs = 1
n_iters = math.ceil(len(eye_dataset)/batch_size)*n_epochs

train_data_loader = DataLoader(eye_dataset, batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # (3x224x224x1) => (3x220x220x32)
        self.clayer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,
                                 stride=1, padding=0) 
        self.relu1 = nn.ReLU()
        # (3x220x220x32) => (3x110x110x32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        # (3x110x110x32) => (3x106x106x62)
        self.clayer2 = nn.Conv2d(in_channels=32, out_channels=62, kernel_size=5,
                                 stride=1, padding=0)
        self.relu2 = nn.ReLU()
        # (3x106x106x62) => (3x53x53x62)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(53*53*62,5)
        
    def forward(self, x):
        out = self.clayer1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        
        out = self.clayer2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        return out
    
    
model = CNN()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_function = nn.CrossEntropyLoss()

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

iter_number = 0
for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_data_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = loss_function(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        
        iter_number += 1
        print('Done',epoch,i)
    for images, labels in train_data_loader:
        total = 0
        correct = 0
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        
        if torch.cuda.is_available():
            correct += (predicted.cpu() == labels.cpu()).sum()
        else:
            correct += (predicted == labels).sum()
    accuracy = 100 * correct / total
    
    print("End of Epoch: {}\t Loss: {}\t Accuracy: {}\t".format(epoch, loss.item(), accuracy))