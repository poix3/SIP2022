import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage.util as su
from scipy import ndimage

#Question 3.1
img = io.imread('eight.tif')
img_sp = su.random_noise(img, mode='s&p')
img_gau = su.random_noise(img, mode='gaussian')
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(131, title='original image')  # left side
# ax1.set_title('mean filtering of salt & pepper')
ax2 = fig.add_subplot(133, title='original image with s&p noise')  # right side
# ax2.set_title('median filtering of salt & pepper')
# ax1.axes.xaxis.set_visible(False)
# ax1.axes.yaxis.set_visible(False)
# ax2.axes.xaxis.set_visible(False)
# ax2.axes.yaxis.set_visible(False)
ax1.imshow(img)
ax2.imshow(img_sp)
# plt.title('mean filtering of salt & pepper')
plt.savefig('ori+sp.png')
# plt.show()
# plt.savefig('eight_sp_filter.tif')
plt.close()

fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(131, title='original image')  # left side
# ax1.set_title('mean filtering of salt & pepper')
ax2 = fig.add_subplot(133, title='original image with gau noise')  # right side
# ax2.set_title('median filtering of salt & pepper')
# ax1.axes.xaxis.set_visible(False)
# ax1.axes.yaxis.set_visible(False)
# ax2.axes.xaxis.set_visible(False)
# ax2.axes.yaxis.set_visible(False)
ax1.imshow(img)
ax2.imshow(img_sp)
# plt.title('mean filtering of salt & pepper')
plt.savefig('ori+gau.png')
# plt.show()
# plt.savefig('eight_sp_filter.tif')
plt.close()


# img_sp_meanf = ndimage.uniform_filter(img_sp, size=3)
# img_sp_medianf = ndimage.median_filter(img_sp, size=3)

# img_gau_meanf = ndimage.uniform_filter(img_gau, size=3)
# img_gau_medianf = ndimage.median_filter(img_gau, size=3)

# fig = plt.figure()
# # plt.gray()  # show the filtered result in grayscale
# ax1 = fig.add_subplot(131, title='mean filtering of salt & pepper')  # left side
# # ax1.set_title('mean filtering of salt & pepper')
# ax2 = fig.add_subplot(133, title='median filtering of salt & pepper')  # right side
# # ax2.set_title('median filtering of salt & pepper')
# ax1.imshow(img_sp_meanf)
# ax2.imshow(img_sp_medianf)
# # plt.title('mean filtering of salt & pepper')
# plt.savefig('3.1(1).png')
# # plt.show()
# # plt.savefig('eight_sp_filter.tif')
# plt.close()


# fig = plt.figure()
# # plt.gray()  # show the filtered result in grayscale
# ax1 = fig.add_subplot(131)  # left side
# # ax1.set_title('mean filtering of salt & pepper')
# ax2 = fig.add_subplot(133)  # right side
# # ax2.set_title('median filtering of salt & pepper')
# ax1.set_title('mean filtering of gaussian')
# ax2.set_title('median filtering of gaussian')
# ax1.imshow(img_gau_meanf)
# ax2.imshow(img_gau_medianf)
# # plt.title('mean filtering of salt & pepper')
# plt.savefig('3.1(2).png')
# # plt.show()
# # plt.savefig('eight_gau_filter.tif')
# plt.close()

# # plt.imshow(img_sp_medianf, cmap='gray')
# # plt.imshow(img, cmap='gray')
# # plt.imshow(img_sp, cmap='gray')
# # plt.imshow(imp_gra, cmap='gray')
# # plt.show()

# import timeit
# from timeit import Timer

# import_module = '''
# import matplotlib.pyplot as plt
# from skimage import io
# import skimage.util as su
# from scipy import ndimage

# img = io.imread('eight.tif')
# img_sp = su.random_noise(img, mode='s&p')
# img_gau = su.random_noise(img, mode='gaussian')

# def test1(N):
#     return ndimage.uniform_filter(img_sp, size=N)

# def test2(N):
#     return ndimage.median_filter(img_sp, size=N)

# def test3(N):
#     return ndimage.uniform_filter(img_gau, size=N)

# def test4(N):
#     return ndimage.median_filter(img_gau, size=N)

# '''

# def name(num):
#     if num == 1:
#         return 'mean filter for s&p'
#     elif num == 2:
#         return 'median filter for s&p'
#     elif num == 3:
#         return 'mean filter for gau'
#     else:
#         return 'median filter for gau'
# for x in [1,3]:
#     plot_y = []
#     for n in range(1, 26):
#         timer = Timer(setup=import_module, stmt='test'+str(x)+'('+str(n)+')')
#         time = timer.timeit(100)
#         plot_y.append(time)
#     plt.plot([i for i in range(1,26)], plot_y, label=name(x))
# plt.legend()
# # print(plot_y)
# plt.xlabel("kernel size N")
# plt.ylabel("computational times")
# plt.savefig('3.1(3).png', facecolor='w')
# # plt.show()
# plt.close()

# for x in [2,4]:
#     plot_y = []
#     for n in range(1, 26):
#         timer = Timer(setup=import_module, stmt='test'+str(x)+'('+str(n)+')')
#         time = timer.timeit(100)
#         plot_y.append(time)
#     plt.plot([i for i in range(1,26)], plot_y, label=name(x))
# plt.legend()
# # print(plot_y)
# plt.xlabel("kernel size N")
# plt.ylabel("computational times")
# plt.savefig('3.1(4).png', facecolor='w')
# # plt.show()
# plt.close()


# # Question 3.2
# def n2t(n, sigma):
#     t = (((n - 1)/2)-0.5)/sigma
#     return t

# fig = plt.figure()
# for n in range(1, 9):
#     img_gauf = ndimage.gaussian_filter(img_sp, sigma=5, truncate=n2t(n**2, 5))
#     ax = fig.add_subplot(2, 4, n, title='n = ' + str(n**2))
#     ax.axes.xaxis.set_visible(False)
#     ax.axes.yaxis.set_visible(False)
#     ax.imshow(img_gauf)
# plt.savefig('3.2(1).png', facecolor='w')
# # plt.show()
# plt.close()

# fig = plt.figure()
# for n in range(1, 9):
#     img_gauf = ndimage.gaussian_filter(img_gau, sigma=5, truncate=n2t(n**2, 5))
#     ax = fig.add_subplot(2, 4, n, title='n = ' + str(n**2))
#     ax.axes.xaxis.set_visible(False)
#     ax.axes.yaxis.set_visible(False)
#     ax.imshow(img_gauf)
# plt.savefig('3.2(2).png', facecolor='w')
# # plt.show()
# plt.close()

# # Question 3.3
# fig = plt.figure()
# for s in range(1, 9):
#     img_gauf = ndimage.gaussian_filter(img_sp, sigma=s, truncate=n2t((3*s)**2, 5))
#     ax = fig.add_subplot(2, 4, s, title='sigma='+str(s)+',n=' + str(3*s))
#     ax.axes.xaxis.set_visible(False)
#     ax.axes.yaxis.set_visible(False)
#     ax.imshow(img_gauf)
# plt.savefig('3.3(1).png', facecolor='w')
# # plt.show()
# plt.close()

# fig = plt.figure()
# for s in range(1, 9):
#     img_gauf = ndimage.gaussian_filter(img_gau, sigma=s, truncate=n2t((3*s)**2, 5))
#     ax = fig.add_subplot(2, 4, s, title='sigma='+str(s)+',n=' + str(3*s))
#     ax.axes.xaxis.set_visible(False)
#     ax.axes.yaxis.set_visible(False)
#     ax.imshow(img_gauf)
# plt.savefig('3.3(2).png', facecolor='w')
# # plt.show()
# plt.close()