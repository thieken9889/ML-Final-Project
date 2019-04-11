'''
Pneumonia predicting algorithm.
Final project for ML 4194.02, SP19:
'''

''' Download the data set and remove the "chest_xray" folder so the path looks 
    like this: .../ML_Final_Project/ML-Final-Project/chest_xray/ '''


import os
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# create transforms for data
# TODO we need to normalize the dataset!!!
nrow = 64 # nrow x nrow image
data_transform = transforms.Compose([transforms.Resize((nrow,nrow)), transforms.ToTensor()])

# create the data set and data loader for train and test data
train_ds = ImageFolder(root='chest_xray\\train', transform=data_transform)
train_dl = DataLoader(train_ds,batch_size=10,shuffle=True)

test_ds = ImageFolder(root='chest_xray\\test', transform=data_transform)
test_dl = DataLoader(test_ds,batch_size=10,shuffle=True)

Xtr, ytr = next(iter(train_dl))


#### Code from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html #####
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0)) # rearrange dimensions from (color,y,x) -> (y,x,color)
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean # undo the normalization
    inp = np.clip(inp, 0, 1)
    # Display image, without ticks
    plt.imshow(inp)
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show() # added this to make the plot show in pycharm

###########################################################################################

imshow(make_grid(Xtr[0:7,:,:,:]))