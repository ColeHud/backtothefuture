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
from colorizer import Colorizer
# from test import Colorizer
import argparse
from dataset import ImageFolder
# from checkpoint import *
from quantize import Bin_Converter
from skimage import color
from torchsummary import summary

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
    parser.add_argument('--data_path', type=str, default='celeba')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-08)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--print_loss_freq', type=int, default=10)
    parser.add_argument('--checkpoint_freq', type=int, default=10)
    return parser.parse_args()


def train(model, trainloader, valloader, num_epoch, optimizer, criterion, args, device): # Train the model
    print("Start training...")
    trn_loss_hist = []
    val_loss_hist = []
    for i in range(num_epoch):
        model.train()
        running_loss = []
        for idx, data in enumerate(trainloader):
            optimizer.zero_grad()
            L_img = data[0].to(device)
            bin_img = data[1].to(device).long()
            pred = model(L_img)
            # pred is NxCxHxW and C is 313
            pred = pred.permute(0, 2, 3, 1).flatten(1, 2)
            pred = pred.flatten(0, 1)
            bin_img = bin_img.permute(0, 2, 3, 1).flatten(1, 2)
            bin_img = bin_img.flatten(0, 1).squeeze()
            # make into one long N*H*WxC vector for both and compare
            loss = criterion(pred, bin_img)
            running_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        model.eval()
        running_val_loss = []
        with torch.no_grad():
            for idx, data in enumerate(valloader):
                L_img = data[0].to(device)
                bin_img = data[1].to(device).long()
                pred = model(L_img)
                pred = pred.permute(0, 2, 3, 1).flatten(1, 2)
                pred = pred.flatten(0, 1)
                bin_img = bin_img.permute(0, 2, 3, 1).flatten(1, 2)
                bin_img = bin_img.flatten(0, 1).squeeze()
                loss = criterion(pred, bin_img)
                running_val_loss.append(loss.item())

        train_loss = np.mean(running_loss)
        val_loss = np.mean(running_val_loss)
        print("\nEpoch {} train loss:{} val loss: {}".format(i + 1, train_loss, val_loss))
        trn_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        if (i+1) % args.checkpoint_freq == 0:
            torch.save({'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        }, 'checkpoints/model_'+str(i+1)+'.pt')
    return trn_loss_hist, val_loss_hist


def init_transform(args):
    # if we flip, then need to flip both
    transform = transforms.Compose([
            transforms.ToTensor()
        ])
    return transform


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
    bin_converter = Bin_Converter()
    dataset = ImageFolder(args.data_path, transform)
    train_set = Subset(dataset, range(4000))
    val_set = Subset(dataset, range(4000, 4500))
    test_set = Subset(dataset, range(4500, 5001))
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size)

    # code to check it loaded properly
    # for idx, data in enumerate(trainloader):
    #     print(idx, data[0].shape)
    #     print(idx, data[1].shape)
    #     L_img = np.squeeze(np.array(data[0][0, :, :, :]))
    #     test = np.array(data[1])
    #     print(test.shape)
    #     test = np.squeeze(test)
    #     test = bin_converter.convert_AB(test[0, :, :])
    #     img = test.transpose(1, 2, 0)
    #     print(img.shape)
    #     img = np.dstack((L_img * 100, img[:, :, 0], img[:, :, 1]))
    #     print(img)
    #     img = (255 * np.clip(color.lab2rgb(img), 0, 1)).astype(np.uint8)
    #     print(img.shape)
    #     plt.imshow(img)
    #     plt.show()

    # Initialize Model
    net = Colorizer().to(device)
    print("Model Summary:")
    summary(net, (1,64,64))

    # # Define loss function, and optimizer
    class_rebalancing = torch.tensor(bin_converter.weights, dtype=torch.float).to(device)
    # print(class_rebalancing)
    # TODO:
    # 1 / lambda+prior
    criterion = torch.nn.CrossEntropyLoss(weight=class_rebalancing)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    # Checkpoints
    #model, start_epoch, stats = restore_checkpoint(model, config("cnn.checkpoint"))
    #TODO: figure out config files, other loop for patience, stats variable
    # Save model parameters
    #save_checkpoint(model, epoch + 1, config("cnn.checkpoint"), stats)

    # Train
    trn_loss_hist, vaL_loss_hist = train(net, trainloader, valloader, args.num_epochs, optimizer, criterion, args, device)
    # Evaluate
    # evaluate(net, testloader)

    #TODO: set up checkpoints (optionally: early stopping, augmentation)


"""
how to run on google collab
https://towardsdatascience.com/google-colab-import-and-export-datasets-eccf801e2971

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
