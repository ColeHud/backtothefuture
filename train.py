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
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, models
from torchvision import transforms as transforms
from torch.utils.data.dataset import Dataset
# from colorizer import Colorizer
import argparse
from dataset import ImageFolder
from checkpoint import *
from quantize import Bin_Converter

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
    # if we flip, then need to flip both
    transform_grey = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize([args.image_size, args.image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5))
        ])
    transform_RGB = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
    return [transform_grey, transform_RGB]

def train(model, trainloader, valloader, num_epoch = 10): # Train the model
  print("Start training...")
  trn_loss_hist = []
  trn_acc_hist = []
  val_acc_hist = []
  model.train() # Set the model to training mode
  for i in range(num_epoch):
    running_loss = []
    print('-----------------Epoch = %d-----------------' % (i+1))
    for batch, label in tqdm(trainloader):
      batch = batch.to(device)
      label = label.to(device)
      optimizer.zero_grad() # Clear gradients from the previous iteration
      pred = model(batch) # This will call Network.forward() that you implement
      loss = criterion(pred, label) # Calculate the loss
      running_loss.append(loss.item())
      loss.backward() # Backprop gradients to all tensors in the network
      optimizer.step() # Update trainable weights
    print("\n Epoch {} loss:{}".format(i+1,np.mean(running_loss)))

    # Keep track of training loss, accuracy, and validation loss
    trn_loss_hist.append(np.mean(running_loss))
    trn_acc_hist.append(evaluate(model, trainloader))
    print("\n Evaluate on validation set...")
    val_acc_hist.append(evaluate(model, valloader))
  print("Done!")
  return trn_loss_hist, trn_acc_hist, val_acc_hist


def evaluate(model, loader):
  model.eval() # Set the model to evaluation mode
  correct = 0
  with torch.no_grad(): # Do not calculate grident to speed up computation
    for batch, label in tqdm(loader):
      batch = batch.to(device)
      label = label.to(device)
      pred = model(batch)
      correct += (torch.argmax(pred,dim=1)==label).sum().item()
    acc = correct/len(loader.dataset)
    print("\n Evaluation accuracy: {}".format(acc))
    return acc


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

    # Load Dataset
    dataset = ImageFolder(args.data_path, transform)
    dataloader =  torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    for idx, data in enumerate(dataloader):
        grey = data[0]
        rgb = data[1]
        plt.imshow(torch.squeeze(grey[0]), cmap='gray')
        plt.show()
        plt.imshow(rgb[0, 0, :, :], cmap='gray')
        plt.show()
        print(grey.shape)

    original_data = torchvision.datasets.ImageFolder(args.data_path, transform)

    # Split data into train, val, test sets
    # Currently 80, 10, 10 - can adjust
    tr = Subset(original_data, range(4000))
    va = Subset(original_data, range(4000, 4500))
    te = Subset(original_data, range(4500, 5001))
    trainloader = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(va, batch_size=args.batch_size, shuffle=False)
    testloader = DataLoader(te, batch_size=args.batch_size, shuffle=False)

    # Initialize Model
    net = Colorizer().to(device)
    print("Model Summary:")
    summary(net, (1,128,128))

    # Define loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss(weights=Bin_Converter.weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Checkpoints
    #model, start_epoch, stats = restore_checkpoint(model, config("cnn.checkpoint"))
    #TODO: figure out config files, other loop for patience, stats variable
    # Save model parameters
    #save_checkpoint(model, epoch + 1, config("cnn.checkpoint"), stats)

    # Train
    trn_loss_hist, trn_acc_hist, val_acc_hist = train(net, trainloader, 
                                                  valloader, args.num_epochs)
    # Evaluate
    evaluate(net, testloader)

    #TODO: set up checkpoints (optionally: early stopping, augmentation)


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
