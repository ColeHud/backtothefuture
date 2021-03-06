import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import color

# palette = np.load("richzhang_palette/pts_in_hull.npy")
# print(palette)
# print(palette.shape)
#
# weights = np.load("richzhang_palette/prior_probs.npy")
# print(weights)
# print(np.sum(weights))

class Bin_Converter():
    def __init__(self):
        self.palette = np.load("/content/drive/MyDrive/EECS442Group/richzhang_palette/pts_in_hull.npy")
        # prior = np.load("/content/drive/MyDrive/EECS442Group/richzhang_palette/prior_probs.npy")

        # add our own counts and divide to get prior probabilities
        prior = np.load("/content/drive/MyDrive/EECS442Group/celeb_counts.npy")
        prior[prior == 0] = 1
        prior = prior / np.sum(prior)

        # self.weights = 0.8 + prior 

        lambda_ = 0.5
        #Apply function for weights
        uniform = np.zeros_like(prior)
        uniform[prior > 0] = 1 / (prior > 0).sum()

        self.weights = 1 / ((1 - lambda_) * prior + lambda_ * uniform)
        self.weights /= np.sum(prior * self.weights)
        # print("weights:", self.weights)

    # image is a numpy array CxHxW
    def convert_bin(self, image):
        # get the l2 difference between binned AB values and bins
        bin_image = np.zeros((image.shape[1], image.shape[2]))

        for x in range(image.shape[1]):
            for y in range(image.shape[2]):
                bin_dists = np.linalg.norm(np.abs(self.palette - image[:, x, y]), axis=1)
                bin_image[x, y] = np.argmin(bin_dists)

        return bin_image

    def convert_AB(self, image):
        AB_image = np.zeros((2, image.shape[0], image.shape[1]))
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                AB_image[0, x, y] = self.palette[image[x, y].astype(np.int64), 0]
                AB_image[1, x, y] = self.palette[image[x, y].astype(np.int64), 1]
        return AB_image

#quantize test data     
if __name__ == "__main__":
    
    test = Bin_Converter()
    """
    #img = plt.imread("/content/drive/MyDrive/EECS442Group/cocostuff-2017/000000000139.jpg")
    img = plt.imread("/content/drive/MyDrive/EECS442Group/celeba/000008.jpg")
    img = color.rgb2lab(img).astype(np.float32)
    L_img = img[:, :, 0]
    print(img[:, :, 1])
    print(img[:, :, 2])
    img = np.dstack((img[:, :, 1], img[:, :, 2]))
    img = img.transpose(2, 0, 1)
    img = test.convert_bin(img)
    img = test.convert_AB(img)
    img = img.transpose(1, 2, 0)
    img = np.dstack((L_img, img[:, :, 0], img[:, :, 1]))
    img = (255 * np.clip(color.lab2rgb(img), 0, 1)).astype(np.uint8)

    print(img.shape)
    plt.imshow(img)
    plt.show()
    """