# backtothefuture
The goal of this project is to implement image colorization using classification. We used a Convolutional Neural Network (CNN) model to perform this classification with various CONV layers using ReLU as an activation function. We trained our model using two datasets: celeb-a and flowers. We read the images in as LAB color and convert them into their respective quantized bins.

Below is a description of each of the folders and files in our github:

zhang_palette: a folder including two numpy files containing prior probabilities and points in hull. These are used to convert to and from quantize bins to LAB color.

.gitignore: a file used to ignore different folders and files when pushing to github. The files we ignore are the datasets and the h5 files which contain the quantized bins.

colorizer.py: a file that contains the architecture for our model under the class name Colorizer.

create_dataset.py: a file that is used upon initialization to create the h5 files so that we don't have to load the large datasets every time.

dataset.py: a file that loads the images from the h5 files to be used for classification.

generate.py: a file used after training to evaluate the performance on our heldout test set. This  process uses SSIM and PSNR as metrics for evaluation. Additionally, we use an average across the x best classifications for a given pixel.

quantize.py: a file that uses the palettes and weights from the Zhang paper in order to convert our images from LAB to binned values.

train.py: a file that sets up the parameters read in from the command line that are used from training. Using these parameters, this file trains our CNN model using the data sets specified earlier. We investigated a variety of different losses: cross entropy without weights specified, cross entropy using the class rebalancing specified in the Zhang paper, and cross entropy with a normalization based off the color distribution for that data set. For all experimentation, we used the Adam optimizer.
