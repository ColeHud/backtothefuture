import numpy as np
import h5py
from PIL import Image
import os
import os.path
from quantize import Bin_Converter
import dataset
import matplotlib.pyplot as plt
from skimage import color


if __name__ == "__main__":
    # hf = h5py.File('data.h5', 'w')
    hf = h5py.File('data.h5', 'r')
    converter = Bin_Converter()

    counter = 0
    images = dataset.make_dataset("cocostuff-2017")
    for image_path in images:
        # img = Image.open(image_path).convert('RGB')
        # img = img.resize((64, 64))
        # img = np.array(img)
        # img = color.rgb2lab(img).astype(np.float32)
        # img = np.dstack((img[:, :, 1], img[:, :, 2]))
        # img = img.transpose(2, 0, 1)
        # img = converter.convert_bin(img)
        # hf.create_dataset(image_path, data=img)

        data = hf.get(image_path)
        arr = np.array(data)
        print(arr)
        print(counter)
        counter += 1


