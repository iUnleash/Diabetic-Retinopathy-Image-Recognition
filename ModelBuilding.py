#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 14:05:33 2019

@author: abdullah
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import confusion_matrix
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler

def train_model(model, criterion, optimizer, #scheduler, 
                num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            #if phase == 'train':
             #   scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def balanceClasses(images,subset_idx, nclasses):                        
    count = [0] * nclasses                                                      
    for i in subset_idx:                                                        
        count[images.imgs[i][1]] += 1                                                    
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                  
    for i in range(nclasses):                                                  
        weight_per_class[i] = N/float(count[i])                                
    weight = [0] * len(subset_idx)                                              
    for i, j in enumerate(subset_idx):                                          
        weight[i] = weight_per_class[images.imgs[j][1]]                                  
    return weight      


path = os.path.join(os.path.expanduser('~'), 'Documents', 'Datasets','aptos2019-blindness-detection','train')
'RGB Mean - 117.416, 62.874, 20.417 - all divided by 255'
'RGB Stdev - 63.443, 35.275, 20.565 - all divided by 255'
'transforms.RandomAffine(degrees = (-360,360), shear = (-45,45), translate=(0,.1)),'

transform = transforms.Compose(
        [transforms.Resize((320,320)),
        #transforms.RandomCrop(),
        #transforms.RandomHorizontalFlip(.33),
        #transforms.RandomVerticalFlip(.33),
        transforms.RandomApply([
        #transforms.ColorJitter(brightness=(1,3), contrast=(1,3), saturation=(1,2)),
        transforms.RandomAffine(degrees = (-360,360), shear = (-45,45))],
        p=.5),
        transforms.ToTensor(),
        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081])
        ])

image_datasets = datasets.ImageFolder(path, transform)
train, val = random_split(image_datasets, (2930, 732))
weights = balanceClasses(image_datasets,train.indices,5)



trainSize = len(train)
valSize = len(val)


weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, trainSize)
trainLoader =  torch.utils.data.DataLoader(train, sampler=sampler,
                                              num_workers=0, batch_size=16)
        
valLoader = torch.utils.data.DataLoader(val, shuffle=True,
                                              num_workers=0, batch_size=16)

dataloaders = {'train': trainLoader,
               'val': valLoader}

dataset_sizes = {'train':trainSize,
                 'val': valSize}

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.457, 0.247, 0.082])
    std = np.array([0.251, 0.140, 0.083])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)  # pause a bit so that plots are updated


inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
class_names = image_datasets.classes
imshow(out, title=[class_names[x] for x in classes])
print(inputs.shape)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.conv1 = nn.Conv2d(3,8,5,1)
        self.conv2 = nn.Conv2d(8,16,5,1)
        self.conv3 = nn.Conv2d(16,24,5,1)
        self.conv4 = nn.Conv2d(24,32,5,1)
        self.conv5 = nn.Conv2d(32,40,5,1)
        self.conv6 = nn.Conv2d(40,48,5,1)
        self.fc1 = nn.Linear(33*33*48,1000)
        self.fc2 = nn.Linear(1000, 5)
    def forward(self, x):
        x = x.view(-1,3,320,320)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 33*33*48)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)
   
   
#model = convNet()

model = torchvision.models.resnet101(pretrained=True)
ct = 0
for child in model.children():
    ct += 1
    if ct < 5:
        for param in child.parameters():
            param.requires_grad = False
            
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr= 3e-5)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, #exp_lr_scheduler,
                       num_epochs=25)


