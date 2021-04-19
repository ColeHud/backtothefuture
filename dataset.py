# custom dataloader modified from https://github.com/junyanz
import torch.utils.data as data

from PIL import Image
import os
import os.path
from skimage import color
import h5py
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.bin_hf = h5py.File('bin_data.h5', 'r')
        self.L_hf = h5py.File('L_data.h5', 'r')
        self.AB_hf = h5py.File('AB_data.h5', 'r')

    def __getitem__(self, index):
        # get the image path
        path = self.imgs[index]

        # retrieve the L channel and binned image
        L_img = self.L_hf.get(path)
        L_img = np.array(L_img)
        bin_img = self.bin_hf.get(path)
        bin_img = np.array(bin_img)

        # apply transforms and return
        L_img = self.transform(L_img)
        bin_img = self.transform(bin_img)
        return L_img, bin_img

    def __len__(self):
        return len(self.imgs)


