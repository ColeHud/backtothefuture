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
import argparse
from dataset import ImageFolder, make_dataset
from quantize import Bin_Converter
from skimage import color
import h5py


if __name__ == "__main__":
    # Set GPU or CPU
    if torch.cuda.is_available():
        print("Using the GPU. You are good to go!")
        device = torch.device('cuda:0')
    else:
        raise Exception("no cuda :(")

    # set epoch of choice
    load_epoch = 20

    bin_converter = Bin_Converter()
    model = Colorizer().to(device)
    checkpoint = torch.load("checkpoints/model_" + str(load_epoch) + ".pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    L_hf = h5py.File('L_data.h5', 'r')
    AB_hf = h5py.File('AB_data.h5', 'r')

    counter = 0
    images = make_dataset("celeba")
    with torch.no_grad():
        for image_path in images:
            img_path = 8823 + counter
            image_path = r"celeba\00" + str(img_path) + ".jpg"
            print(image_path)
            img = Image.open(image_path).convert('RGB')
            img = img.resize((64, 64), Image.BICUBIC)
            img = np.array(img)
            img = color.rgb2lab(img).astype(np.float32)

            L_img = np.array(L_hf.get(image_path))
            # L_img = L_img * 100
            AB_img = AB_hf.get(image_path)

            transform = transforms.ToTensor()
            input = transform(L_img).to(device)
            input = torch.unsqueeze(input, 0)
            pred = model(input)
            pred = np.squeeze(pred.cpu().numpy())
            # print(pred.shape)
            # first = np.argmax(pred, axis=0)
            # # print(first)
            # pred[first] = -100000
            # second = np.argmax(pred, axis=0)
            # pred[second] = -100000
            # third = np.argmax(pred, axis=0)
            # pred[third] = -100000
            # fourth = np.argmax(pred, axis=0)
            # pred[fourth] = -100000
            # fifth = np.argmax(pred, axis=0)

            pred_AB = []
            num_max_bins = 4
            for i in range(num_max_bins):
                max = np.argmax(pred, axis=0)
                pred[max] = -100000
                pred_AB.append(bin_converter.convert_AB(max))

            # pred_AB.append(bin_converter.convert_AB(first))
            # pred_AB.append(bin_converter.convert_AB(second))
            # pred_AB.append(bin_converter.convert_AB(third))
            # pred_AB.append(bin_converter.convert_AB(fourth))
            # pred_AB.append(bin_converter.convert_AB(fifth))

            test = np.dstack((img[:, :, 0], pred_AB[0].transpose(1,2,0)))
            test = (255 * np.clip(color.lab2rgb(test), 0, 1)).astype(np.uint8)
            plt.imshow(test)
            plt.show()

            pred_AB = np.mean(np.array(pred_AB), axis=0)
            print(pred_AB.shape)
            pred_AB = pred_AB.transpose(1, 2, 0)

            print(L_img.shape, pred_AB.shape)
            output = np.dstack((img[:, :, 0], pred_AB))
            output = (255 * np.clip(color.lab2rgb(output), 0, 1)).astype(np.uint8)

            plt.imshow(output)
            plt.show()
            counter += 1


