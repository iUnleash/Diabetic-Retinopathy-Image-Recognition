#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 14:05:33 2019

@author: abdullah
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import cv2
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler

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
transform = transforms.Compose(
        [transforms.Resize((320,320)),
        #transforms.RandomCrop(),
        transforms.RandomHorizontalFlip(.33),
        transforms.RandomVerticalFlip(.33),
        #transforms.ColorJitter(brightness=.5, contrast=.2, saturation=1, hue=.35),
        #transforms.RandomAffine(6),
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
                                              num_workers=0, batch_size=10)
        
valLoader = torch.utils.data.DataLoader(val, shuffle=True,
                                              num_workers=0, batch_size=10)


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

imshow(out)
print(inputs.shape)


