## <b> Back to the Future: Image Colorization </b> ##
The goal of this project is to implement image colorization using classification. We used a Convolutional Neural Network (CNN) model to perform this classification with various CONV layers using ReLU as an activation function. We trained our model using two datasets: celeb-a and flowers. We read the images in as LAB color and convert them into their respective quantized bins.

The majority of this implementation was from scratch, however ideas were borrowed from Zhang et al.'s paper for the architecture structure and loss components. Additionally, we borrowed weighting code from Time0o's github to add to convert weights to priors. Finally, we built on top of Junyanz' Pix2Pix dataloader to produce L and quantized binned outputs for each image. All sources are highlighted in citations and commented in our code.

**Clone the repository; install datasets separately**

```
git clone https://github.com/ColeHud/backtothefuture.git
```

Note: to preprocess data, checkout create_dataset.py and run on the downloaded dataset before running training. The pathnames stored in the h5 file could change depending on Windows or Linux with the forward/backslash


**Colorize: Example of running code to colorize images with the given path**
```
python train.py --data_path 'celeba' --image_size 128 --num_epochs 50 --batch_size 8 --lr 1e-4 --weight_decay 1e-4
```

**Example Output**

<img src="https://github.com/ColeHud/backtothefuture/blob/main/bttf.gif" width="200">

Above is an example of our model output merging different colorized frames from the movie "Back to the Future". This image displays the results of taking black and white images from the original movie, feeding them into our model which outputs the colorized version.

**Description of Files**

**zhang_palette**: a folder including two numpy files containing prior probabilities and points in hull. These are used to convert to and from quantize bins to LAB color.

**.gitignore**: a file used to ignore different folders and files when pushing to github. The files we ignore are the datasets and the h5 files which contain the quantized bins.

**colorizer.py**: a file that contains the architecture for our model under the class name Colorizer.

**colorizer.ipynb**: a file containing code from colorizer.py, generate.py, and train.py which can fully load the data from the h5 files, train, and evaluate the model in Google Colab.

**create_dataset.py**: a file that is used upon initialization to create the h5 files so that we don't have to load the large datasets every time.

**dataset.py**: a file that loads the images from the h5 files to be used for classification.

**generate.py**: a file used after training to evaluate the performance on our heldout test set. In the python file, it visualizes the image. On the fully trained model in our ipynb, we also gather SSIM and PSNR as metrics for evaluation. Additionally, we use an average across the top-k best classifications for a given pixel.

**quantize.py**: a file that uses the palettes and weights from the Zhang paper in order to convert our images from LAB to binned values.

**train.py**: a file that sets up the parameters read in from the command line that are used from training. Using these parameters, this file trains our CNN model using the data sets specified earlier. We investigated a variety of different losses: cross entropy without weights specified, cross entropy using the class rebalancing specified in the Zhang paper, and cross entropy with a normalization based off the color distribution for that data set. For all experimentation, we used the Adam optimizer.

### Citations ###
```
@article{junyanz,
  title={pytorch-CycleGAN-and-pix2pix},
  author={junyanz},
  year={2020}
}

@article{Time0o,
  title={pytorch-colorful-colorization},
  author={Time0o},
  year={2020}
}

@article{Cheng,
  title={Deep colorization. In: Proceedings of the IEEE International Conference on Computer Vision},
  author={Z. Cheng and Q. Yang and B. Sheng},
  year={2015},
  pages={415--423}

}

@inproceedings{zhang2016colorful,
  title={Colorful Image Colorization},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A},
  booktitle={ECCV},
  year={2016}
}

@article{zhang2017real,
  title={Real-Time User-Guided Image Colorization with Learned Deep Priors},
  author={Zhang, Richard and Zhu, Jun-Yan and Isola, Phillip and Geng, Xinyang and Lin, Angela S and Yu, Tianhe and Efros, Alexei A},
  journal={ACM Transactions on Graphics (TOG)},
  volume={9},
  number={4},
  year={2017},
  publisher={ACM}
}
```
