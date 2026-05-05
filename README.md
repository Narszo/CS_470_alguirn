# CS 470: Computer Vision and Image Processing
***Spring 2026***  
***Author: Noah Alguire***  
***Original Author: Dr. Michael J. Reale***  
***SUNY Polytechnic Institute*** 

## Runnable Python Scripts

### BasicVision.py
A basic sample that loads up the relevant libraries, prints versions numbers, and either 1) loads an image from a path specified on the command line, or 2) opens a webcam.
Image(s) will be displayed until a key is hit.
## test

### A01.py
Several grayscale intensity transformations that are accessed through a Gradio web interface.
The Gradio interface allows Histogram stretching, Log, Gamma, Histogram Equalization, and Piecewise transformations

### A02.py:
Image convolution with and without loops as well as a Fourier filter.
It launches a Gradio interface where users can upload a grayscale image and a kernel file,
adjust alpha and beta values, and view the filtered output interactively.

### A03.py:
Uses color segmentation to detect WBC bounding boxes.
SLIC superpixels group the image into color-coherent regions, then k-means clusters the superpixel mean colors.
The cluster closest to blue is selected, morphological operations clean the resulting mask, and connected components extract bounding boxes.

### A03.py:
Image classification using two convolutional neural network architectures trained and evaluated with PyTorch.
SimpleCNN: A basic CNN with three convolutional blocks (Conv → BatchNorm → ReLU → MaxPool) using 32/64/128 filters, followed by two fully-connected layers with Dropout. No data augmentation.
ResCNN: A deeper residual CNN with skip connections and ELU activations. Uses three residual stages with 64/128/256 filters, adaptive average pooling, and random horizontal flip + random crop data augmentation during training.