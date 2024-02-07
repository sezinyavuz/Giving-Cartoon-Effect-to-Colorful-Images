from PIL import Image, ImageFilter, ImageChops
import numpy as np
from scipy import ndimage
from skimage.color import rgb2lab, lab2rgb


image = Image.open(r"../report/{folder}/{file}.jpg")

# If the alpha channel exists remove it
if image.mode == 'RGBA':
    image = image.convert('RGB')



#PART1 IMAGE SMOOTHING
def image_smoothing(i_image, sigma):
    i_image_array = np.array(i_image)
    red_channel, green_channel, blue_channel = i_image_array[:, :, 0], i_image_array[:, :, 1], i_image_array[:, :, 2]

    # amount of smoothing
    smoothed_red_channel = ndimage.gaussian_filter(red_channel, sigma=sigma)
    smoothed_green_channel = ndimage.gaussian_filter(green_channel, sigma=sigma)
    smoothed_blue_channel = ndimage.gaussian_filter(blue_channel, sigma=sigma)

    # Merge the smoothed channels to  RGB image
    smoothed_image_array = np.stack((smoothed_red_channel, smoothed_green_channel, smoothed_blue_channel), axis=-1)

    output_image = Image.fromarray((smoothed_image_array).astype('uint8'))
    return  output_image


# PART 2  EDGE DETECTION PART

def edge_detection(input_image,sigma1,sigma2,threshold):

    gray_image = input_image.convert('L')

    smoothed_image_1 = gray_image.filter(ImageFilter.GaussianBlur(sigma1))
    smoothed_image_2 = gray_image.filter(ImageFilter.GaussianBlur(sigma2))
    dog = ImageChops.difference(smoothed_image_1, smoothed_image_2)

    def apply_threshold(pixel_value, threshold):
        return 0 if pixel_value >= threshold else 255

    def threshold_image(image, threshold):
        # Apply the threshold function to each pixel in the image
        return image.point(lambda x: apply_threshold(x, threshold))

    edges = threshold_image(dog, threshold)
    return edges


#  PART3 IMAGE QUANTIZATION PART
def image_quantization(input_image, x, y):
    lab_image = rgb2lab(input_image)
    quantized_lab_image = lab_image.copy()
    quantized_lab_image[:, :, 0] = (np.round(quantized_lab_image[:, :, 0] / x) * y)

    quantized_rgb = lab2rgb(quantized_lab_image)

    quantized_rgb_array = np.clip(quantized_rgb * 255, 0, 255).astype('uint8')

    quantized_rgb = Image.fromarray(quantized_rgb_array)

    return quantized_rgb


# PART4  Combining Edge and Quantized Image Part

def combine_images(edges,quantized):
    #inverse of the sketch image
    inverted = edges.convert('RGB')

    # multiply the inverse edges with the quantized image
    combined_image = ImageChops.multiply(inverted, quantized)

    return combined_image



smoothed_image= image_smoothing(image,5)
smoothed_image.show()
smoothed_image.save("ou1.jpg")


edge_detect=edge_detection(image,0.5,5,15)
edge_detect.show()
edge_detect.save("ou2.jpg")

quantized_image=image_quantization(image,10,20)
quantized_image.show()
quantized_image.save("ou3.jpg")

combined_image=combine_images(edge_detect,quantized_image)
combined_image.show()
combined_image.save("ou4.jpg")