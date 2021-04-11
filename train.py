import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import png
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from PIL import Image
import torchvision
from colormap.colors import Color, hex2rgb
from sklearn.metrics import average_precision_score as ap_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from colorizer import Colorizer

#from google colab? idk
if torch.cuda.is_available():
    print("Using the GPU. You are good to go!")
    device = torch.device('cuda:0')
else:
    raise Exception("no cuda :(")



#load dataset
#https://pytorch.org/vision/stable/datasets.html


#model
net = Colorizer().to(device)
print("Model Summary:")
summary(net, (1,128,128))

#train that shit 😤
