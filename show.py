import os, sys
import time
import shutil
import math
import pickle
import numpy as np
import torch
import torch.autograd as A
import torch.nn as nn
from PIL import Image
import matplotlib.image as mpimg
import os
from torchvision import transforms
import torch.nn.functional as F
def get_images():
    root_dir = '/data2/simingy/model_data/sparse_coding/train'

    image_list = os.listdir(root_dir)
    images = list()
    for f in image_list:
        fullname = os.path.join(root_dir, f)
        img = Image.open(fullname).convert('RGB') 
        images.append(img)
    return images

def PIL2array(img):
        return np.array(img.getdata(),
                        np.uint8).reshape(img.size[1], img.size[0], 3)

def train_it(model, criterion, optimizer):
    model.train()
    mean = np.array([0.49139968, 0.48215827, 0.44653124])
    std = np.array([0.24703233, 0.24348505, 0.26158768])
    normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    input_x = get_images()

    #x = torch.from_numpy(x)
    y = np.zeros((2, 1))
    y = torch.from_numpy(y).long()
    
    input_y = list()
    input_x_ = list()
    for i in range(int(math.ceil(float(len(input_x)) / 2))):
        input_y.append(y)
        input_x_.append(input_x[i*2:(i+1)*2])
    
    for i in range(20):    
        for (images,y) in zip(input_x_,input_y):
        
            x = A.Variable(torch.stack([transform(img) for img in images]).cuda())
            y = A.Variable(y.cuda())

            pred = model(x)
            loss = criterion(pred, y.squeeze())
            #print("train loss:", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train_it_one(model, criterion, optimizer):
    model.train()
    mean = np.array([0.49139968, 0.48215827, 0.44653124])
    for i in range(3):
        mean = np.expand_dims(mean, axis=0)
    print("mean:", mean.shape)
    std = np.array([0.24703233, 0.24348505, 0.26158768])
    for i in range(3):
        std = np.expand_dims(std, axis=0)

    # Familiar images
    train_x =  Image.open('/data2/simingy/model_data/sparse_coding/train/input-11.png').convert('RGB')
    x = PIL2array(train_x)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    x = (x - mean) / std
    x = np.transpose(x, (0, 3, 1, 2))
    #x = np.tile(x, (2, 1, 1, 1))
    print("x:", x.shape)
    x = torch.from_numpy(x).float()
    y = np.zeros((1, 1))
    y = torch.from_numpy(y).long()
    
    input_y = list()
    input_x = list()
    for i in range(20):
        input_y.append(y)
        input_x.append(x)
    
 
    for (x,y) in zip(input_x,input_y):
        
        x = A.Variable(x.cuda())
        y = A.Variable(y.cuda())

        pred = model(x, print_fe=1)
        loss = criterion(pred, y.squeeze())
        #print("train loss:", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_it(model, criterion, optimizer):

    mean = np.array([0.49139968, 0.48215827, 0.44653124])
    for i in range(3):
        mean = np.expand_dims(mean, axis=0)
    print("mean:", mean.shape)
    std = np.array([0.24703233, 0.24348505, 0.26158768])
    for i in range(3):
        std = np.expand_dims(std, axis=0)

    # Familiar images
    train_x =  Image.open('/data2/simingy/model_data/sparse_coding/train/input-11.png').convert('RGB')
    x = PIL2array(train_x)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    x = (x - mean) / std
    x = np.transpose(x, (0, 3, 1, 2))
    #x = np.tile(x, (2, 1, 1, 1))
    print("x:", x.shape)
    x = torch.from_numpy(x).float()
    y = np.zeros((1, 1))
    y = torch.from_numpy(y).long()
    
    input_y = list()
    input_x = list()
    for i in range(2):
        input_y.append(y)
        input_x.append(x)
    
    
    for (x,y) in zip(input_x,input_y):
        
        x = A.Variable(x.cuda())
        y = A.Variable(y.cuda())

        pred = model(x, print_fe=1)
        loss = criterion(pred, y.squeeze())
        print("test loss:", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("-----------------------Test---------------------")
    # Novel images
    test_x =  Image.open('/data2/simingy/model_data/sparse_coding/train/input-16.png').convert('RGB')
    x = PIL2array(test_x)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    x = (x - mean) / std
    x = np.transpose(x, (0, 3, 1, 2))
    #x = np.tile(x, (2, 1, 1, 1))
    print("x:", x.shape)
    x = torch.from_numpy(x).float()
    y = np.zeros((1, 1))
    y = torch.from_numpy(y).long()
    
    input_y = list()
    input_x = list()
    for i in range(2):
        input_y.append(y)
        input_x.append(x)
    
    
    for (x,y) in zip(input_x,input_y):
        
        x = A.Variable(x.cuda())
        y = A.Variable(y.cuda())

        pred = model(x, print_fe=1)
        loss = criterion(pred, y.squeeze())
        print("test loss:", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
