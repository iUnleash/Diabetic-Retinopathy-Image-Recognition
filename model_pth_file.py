# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:59:23 2019

@author: nickn
"""

## Code to save model pth file for Kaggle

import os
if os.getcwd() != r"D:\OR610-Project":
    os.chdir(r"D:\OR610-Project")

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
import os
from PIL import Image
import glob
import cv2
import gc #garbage collector for gpu memory 
from tqdm import tqdm


path = r"D:\OR610-Project\train"

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
                                              num_workers=4, batch_size=16)
        
valLoader = torch.utils.data.DataLoader(val, shuffle=True,
                                              num_workers=4, batch_size=16)


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


inputs, classes = next(iter(trainLoader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
class_names = image_datasets.classes
imshow(out, title=[class_names[x] for x in classes])
print(inputs.shape)



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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    tqdm()
    for i, (inputs, labels) in enumerate(tqdm(data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        num_examples += labels.size(0)
        correct_pred += (preds == labels).sum()
    return correct_pred.float()/num_examples * 100


num_epochs = 2
start_time = time.time()
best_acc = 0.0
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= 3e-6)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
losses = []
valid_acc_list = []


for epoch in range(num_epochs):
    print('Seconds elapsed: ', round((time.time() - start_time),2))
    print('Running Epoch: ', epoch+1)
    print('-' * 10)
    running_loss = 0.0
    model.train()
    i = 0
    for iteration, (inputs, labels) in enumerate(trainLoader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item())
        del inputs, labels
        i+=1
                
        # printing the results every 20 iterations
        if not iteration%20:
            print('Epoch {:03d}/{:03d} | Batch: {:03d}/{:03d} |'
                  ' Cost: {:.4f} | Avg Loss: {:.4f}'.format(epoch+1, num_epochs, iteration+1, len(trainLoader),loss, running_loss/i))
            running_loss = 0.0
            i = 0
            
        del loss
        gc.collect()
        torch.cuda.empty_cache()
        
            
    losses.append(running_loss)
    with torch.set_grad_enabled(False):
        model.eval()
        train_accuracy = compute_accuracy(model, trainLoader, device)
        valid_accuracy = compute_accuracy(model, valLoader, device)
        valid_acc_list.append(valid_accuracy)
        print('Epoch: {:03d}/{:03d} Train Acc.: {:.2f} | Validation Acc.: {:.2f}'.format(epoch+1,num_epochs, train_accuracy,
                                                                                valid_accuracy))
    if valid_acc_list[-1] > best_acc:
        best_acc = valid_acc_list[-1]
        torch.save(model.state_dict(), 'train_valid_best.pth')
    # early stopping condition
    if epoch+1 >= 5:
        # if the accuracy is lower than lowest of last 4 values
        if valid_acc_list[-1] < min(valid_acc_list[-5:-1]): 
            print('...Stopping Early...')
            break
        
print("Training Complete --- {} seconds ---".format(round((time.time() - start_time),2)))

