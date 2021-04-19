import numpy as np
import h5py
from PIL import Image
import os
import os.path
import quantize
import dataset


if __name__ == "__main__":
    hf = h5py.File('data.h5', 'w')
    converter = Bin_Converter()

    images = make_dataset("cocostuff-2017")
    for image in images:
        img = plt.imread(image)
        img = color.rgb2lab(img).astype(np.float32)
        img = np.dstack((img[:, :, 1], img[:, :, 2]))
        img = img.transpose(2, 0, 1)
        img = test.convert_bin(img)
        print(img.shape)
