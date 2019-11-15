#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 20:07:40 2019

@author: abdullah
"""

import os
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
from torchvision import datasets, models, transforms


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
    
    



path = os.path.join(os.path.expanduser('~'), 'Documents', 'Datasets','aptos2019-blindness-detection','train')

container = []
for file in glob.glob(path+'/4/*.png'):
    image = cv2.imread(file, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = crop_image_from_gray(image)
    image = cv2.resize(image,(320,320))    
    container.append(image)

image = np.array(container)
cv2.mean(image[:,:,:,2])
cv2.meanStdDev(image[:,:,:,0])
test = image/255

plt.imshow(image[2])