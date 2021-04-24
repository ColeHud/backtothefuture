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
    # EXPERIMENTAL
    mse_crit = nn.MSELoss()
    priors = torch.tensor(np.load("/content/drive/MyDrive/EECS442Group/richzhang_palette/prior_probs.npy")).to(device)

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
            # loss = criterion(pred, bin_img)
            # EXPERIMENTAL
            ce_loss = criterion(pred, bin_img)
            # pred = torch.argmax(pred, axis=1)
            # counts = torch.bincount(pred, minlength=313)
            # counts = counts / torch.sum(counts)
            counts = torch.sum(pred, axis=0, dtype=float)
            counts /= torch.sum(counts)
            mse_loss = mse_crit(counts, priors)
            loss = 0.6*ce_loss + 0.4 * mse_loss
            # END EXPERIMENTAL
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
        print("\nEpoch {} train loss: {} val loss: {}".format(i + 1, train_loss, val_loss))
        trn_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        #SAVE CHECKPOINT every 10 iterations to EECS442Group in gdrive
        """
        if (i % 10) == 0:
          save_checkpoint(i, model, np.mean(running_loss))
        """
        if (i+1) % args['checkpoint_freq'] == 0:
          # CHANGE
          experiment_number = '5'
          torch.save({'epoch': i,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': train_loss,
                      }, '/content/drive/MyDrive/EECS442Group/checkpoints/model_'+str(i+1)+'_'+experiment_number+'.pt')
        #model_name should be model_epoch_idx.pt
        
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

def save_checkpoint(i,net,loss):
    EPOCH = i
    PATH = "model"+str(i)+".pt"
    LOSS = loss

    torch.save({
                'epoch': EPOCH,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': LOSS,
                }, PATH)


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

    # Initialize Model
    net = Colorizer().to(device)
    print("Model Summary:")
    summary(net, (1,64,64))

    #USE THIS IN THE EVENT OF A CRASH TO RETRIEVE MODEL
    """
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    """

    # Define loss function, and optimizer
    class_rebalancing = torch.tensor(bin_converter.weights, dtype=torch.float).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_rebalancing)
    optimizer = torch.optim.Adam(net.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

    # Train
    trn_loss_hist, val_loss_hist = train(net, trainloader, valloader, args['num_epochs'], optimizer, criterion, args, device)
