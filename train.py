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
from torchvision import datasets, models
from torchvision import transforms as transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from colorizer import Colorizer
import argparse
from dataset import ImageFolder

"""
# Set up optimization hyperparameters
learning_rate = 1e-2
weight_decay = 1e-4
num_epoch = 5  # TODO: Choose an appropriate number of training epochs
optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                       weight_decay=weight_decay)
"""

def init_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='cocostuff-2017/')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-08)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--print_loss_freq', type=int, default = 10)
    return parser.parse_args()

def init_transform(args):
    transform_grey = transforms.Compose([
            transforms.Resize(args.image_size), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
    transform_RGB = transforms.Compose([
            transforms.Resize(args.image_size), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
    return [transform_grey, transform_RGB]

if __name__ == "__main__":
    # Set GPU or CPU
    if torch.cuda.is_available():
        print("Using the GPU. You are good to go!")
        device = torch.device('cuda:0')
    else:
        raise Exception("no cuda :(")

    # Grab hyperparameters from command line
    args = init_arguments()
    transform = init_transform(args)

    dataset = ImageFolder(args.data_path, transform)
    dataloader =  torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    for idx, data in enumerate(dataloader):
        grey = data[0]
        rgb = data[1]
        print(grey.shape)

    original_data = torchvision.datasets.ImageFolder(args.data_path, transform)

    
    # Initialize Model
    net = Colorizer().to(device)
    print("Model Summary:")
    summary(net, (1,128,128))

    # Define loss function, and optimizer
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    """

    # Train


class CustomDatasetRGB(torchvision.Dataset):
    def __init__(self, images):
        self.images = images

    def __getitem__(self, index):
        rgb_image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        rgb_image = np.transpose(rgb_image, axes=(2, 0, 1))
        return torch.from_numpy(rgb_image), torch.from_numpy(gray_image)

    def __len__(self):
        return len(self.images)


#load dataset
#https://pytorch.org/vision/stable/datasets.html

#model
net = Colorizer().to(device)
print("Model Summary:")
summary(net, (1,128,128))

# define loss function, and optimizer
"""
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
"""

#optional implementation details: checkpoints, early stopping (both increase speed)

#train that shit 😤





"""
# Helper function
def train_epoch(data_loader, model, criterion, optimizer):
    for i, (X, y) in enumerate(data_loader):
        inputs, labels = X, y

        optimizer.zero_grad()

        outputs = model(inputs) # Forward Pass
        loss = criterion(outputs, labels) # Loss
        loss.backward() # Backward Pass
        optimizer.step() # Update Weights
"""

"""
patience = 5
curr_patience = 0
#

# Loop over the entire dataset multiple times
# for epoch in range(start_epoch, config('cnn.num_epochs')):
epoch = start_epoch
while curr_patience < patience:
    # Train model
    train_epoch(tr_loader, model, criterion, optimizer)

    # Evaluate model
    evaluate_epoch(
        axes, tr_loader, va_loader, te_loader, model, criterion, epoch + 1, stats
    )

    # Save model parameters
    #save_checkpoint(model, epoch + 1, config("cnn.checkpoint"), stats)

    # update early stopping parameters
    #curr_patience, prev_val_loss = early_stopping(
    #    stats, curr_patience, prev_val_loss
    #)

    epoch += 1
"""
