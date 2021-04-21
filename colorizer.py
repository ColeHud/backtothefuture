import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision
from sklearn.metrics import average_precision_score as ap_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

# helper function for Zhang et al. 2016 architecture structure - 2 CONV layers
def make_block_2conv(layers, in_channels, out_channels):
    layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(out_channels, out_channels, 3, 2, 1))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.BatchNorm2d(out_channels))

# helper function for Zhang et al. 2016 architecture structure - 3 CONV layers
def make_block_3conv(layers, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, dilation=dilation))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, dilation=dilation))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation=dilation))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.BatchNorm2d(out_channels))

class Colorizer(nn.Module):
    def __init__(self):
        super(Colorizer, self).__init__()

        self.layers = []
        # in_channels, out_channels, kernel, stride, padding

        make_block_2conv(self.layers, 1, 32)
        make_block_2conv(self.layers, 32, 64)
        make_block_3conv(self.layers, 64, 128, 3, stride=2)
        make_block_3conv(self.layers, 128, 256, 3)
        make_block_3conv(self.layers, 256, 256, 3, padding=2, dilation=2)
        make_block_3conv(self.layers, 256, 256, 3)

        self.layers.append(nn.ConvTranspose2d(256, 128, 4, 2, 1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.ConvTranspose2d(128, 128, 4, 2, 1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.ConvTranspose2d(128, 64, 4, 2, 1))
        self.layers.append(nn.ReLU(inplace=True))

        self.layers.append(nn.Conv2d(64, 313, 1, 1, 0))

        # add softmax and upsample stuff

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)

        # softmax stuff

        return x