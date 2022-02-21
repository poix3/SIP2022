import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, hsv2rgb
from skimage.util import img_as_float

def gamma_transform(image, gamma, c = 1):
    image = img_as_float(image)
    return np.array(c*image**gamma)


# 1.1
img = imread('lena.png', as_gray=True)
imshow(gamma_transform(img, 0.3)) # gamma = [0.3, 0.5, 0.8, 2.0, 3.0]

# 1.2
autumn = imread('autumn.tif')
imshow(autumn)
imshow(gamma_transform(autumn, 0.3)) # gamma = [0.3, 0.5, 0.8, 2.0, 3.0]

# 1.3
hsv_img = rgb2hsv(autumn)
value_img = hsv_img[:, :, 2]
imshow(value_img)
value_img_gamma = gamma_transform(value_img, 3.0) # gamma = [0.3, 0.5, 0.8, 2.0, 3.0]
hsv_img[:, :, 2] = value_img_gamma
imshow(value_img_gamma)
imshow(hsv2rgb(hsv_img))