# Giving-Cartoon-Effect-to-Colorful-Images
The goal of this assignment is to implement a simplified version of Real-Time Video Abstraction[1] by using simple image filters.
#####################Image Processing Code README#################################

# Introduction
This Python code performs image processing tasks, including image smoothing, edge detection, image quantization, and combining edges with quantized images. It utilizes the Python Imaging Library (PIL), NumPy, SciPy, and scikit-image libraries.

# Requirements
- Python 3.x
- Pillow (PIL) library
- NumPy library
- SciPy library
- scikit-image library

# Usage
1. Install required libraries:
    ```bash
    pip install pillow numpy scipy scikit-image
    ```
2. Replace the placeholder in the image file path (`"../report/{folder}/{file}.jpg"`) with the desired input image file path.
3. Run the code to execute the image processing tasks.

# Functions

1. Image Smoothing

image_smoothing(input_image, sigma)
Smooths the input RGB image using Gaussian filtering.

input_image: Input RGB image (PIL Image object).
sigma: Amount of smoothing (standard deviation of the Gaussian filter).

2. Edge Detection

edge_detection(input_image, sigma1, sigma2, threshold)
Performs edge detection using the Difference of Gaussians (DoG) method.

input_image: Input RGB image (PIL Image object).
sigma1: Standard deviation of the first Gaussian kernel.
sigma2: Standard deviation of the second Gaussian kernel.
threshold: Edge detection threshold.

3. Image Quantization

image_quantization(input_image, x, y)
Quantizes the input image in the LAB color space.

input_image: Input RGB image (PIL Image object).
x: Divisor for LAB channel quantization.
y: Multiplier for LAB channel quantization.

4. Combining Edge and Quantized Images

combine_images(edges, quantized)
Combines detected edges with the quantized image.

edges: Image containing detected edges (PIL Image object).
quantized: Quantized image (PIL Image object).
