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
    # bin_hf = h5py.File('bin_data.h5', 'w')
    # L_hf = h5py.File('L_data.h5', 'w')
    # AB_hf = h5py.File('AB_data.h5', 'w')
    bin_hf = h5py.File('bin_data.h5', 'r')
    L_hf = h5py.File('L_data.h5', 'r')
    AB_hf = h5py.File('AB_data.h5', 'r')
    converter = Bin_Converter()

    counter = 0
    images = dataset.make_dataset("cocostuff-2017")
    for image_path in images:
        # Convert to LAB
        # img = Image.open(image_path).convert('RGB')
        # img = img.resize((128, 128), Image.BICUBIC)
        # img = np.array(img)
        # img = color.rgb2lab(img).astype(np.float32)
        #
        # # # Save dataset of normalized L channel
        # L_img = img[:, :, 0]
        # L_img = L_img / 100
        # L_hf.create_dataset(image_path, data=L_img)
        #
        # # # Save dataset of AB channel
        # A_img = (img[:, :, 1] + 86.185) / 184.439
        # B_img = (img[:, :, 2] + 107.863) / 202.345
        # AB_img = np.dstack((A_img, B_img))
        # AB_hf.create_dataset(image_path, data=AB_img)
        #
        # # # Save dataset of binned images
        # binned_AB = np.dstack((img[:, :, 1], img[:, :, 2]))
        # binned_AB = binned_AB.transpose(2, 0, 1)
        # bin_img = converter.convert_bin(binned_AB)
        # bin_hf.create_dataset(image_path, data=bin_img)



        # print(image_path)
        # data = L_hf.get(image_path)
        # print(data)
        # data2 = bin_hf.get(image_path)
        # print(data2)
        # arr = np.array(data)
        # arr2 = np.array(data2)
        # print(arr, arr2)
        print(counter)
        counter += 1


