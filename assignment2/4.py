#Question 4.2
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage.util as su

img = io.imread('eight.tif')
img_sp = su.random_noise(img, mode='s&p')
img_gau = su.random_noise(img, mode='gaussian')

# 4.2
def f(x, y, sigma):
    return math.exp(-(x**2+y**2)/(2*(sigma**2)))

def g(u, tau):
    return math.exp(-(u**2)/(2*(tau**2)))
def w(img, x, y, i, j, sigma, tau):
    return f(i, j, sigma)*g(img[x+i][y+j]-img[x][y], tau)

def filter_1pixel(img,x,y,size,sigma,tau):
    denmonator = 0
    numerator = 0
    for i in range(-size, size+1):
        for j in range(-size, size+1):
            w_result = w(img, x, y, i, j, sigma, tau)
            numerator += w_result*img[x+i][y+j]
            denmonator += w_result
    return numerator/denmonator

def bilateralFilter(img,k,sigma,tau):
    w,h= img.shape
    size = k // 2

    # padding process for boundary pixels
    _img = np.zeros((w+2*size,h+2*size), dtype=float)
    _img[size:size+w,size:size+h] = img.copy()
    dst = _img.copy()

    #Filtering process
    for x in range(w):
        for y in range(h):
            dst[x+size,y+size] = filter_1pixel(_img, x+size, y+size, size, sigma, tau)

    dst = dst[size:size+w,size:size+h]

    return dst

# 4.3

fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
for s in range(1, 9):
    if s != 1:
        img_gauf = bilateralFilter(img_gau, 3, s**2, 2*s)
        ax = fig.add_subplot(2, 4, s, title='σ='+str(s**2)+',τ='+str(2*s))
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.imshow(img_gauf)
    else:
        ax = fig.add_subplot(2, 4, s, title='original gau noise')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.imshow(img_gau)
plt.savefig('4.3.png')
# plt.show()
plt.close()