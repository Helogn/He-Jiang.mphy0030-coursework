import numpy as np
# from numpy.lib.nanfunctions import nanprod
from skimage.measure import marching_cubes
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot  as plt

image = Image.open('SKY.png')
data = np.array(image)
data = data[:,:,1]
Filtered_image = gaussian_filter(data, sigma=2)
Filtered_image1 = gaussian_filter(data, sigma=6)
Filtered_image2 = gaussian_filter(data, sigma=10)
plt.subplot(2,2,1)
# plt.colorbar()
plt.imshow(data)
plt.subplot(2,2,2)
plt.imshow(Filtered_image)
# plt.colorbar
plt.subplot(2,2,3)
plt.imshow(Filtered_image1)
plt.subplot(2,2,4)
plt.imshow(Filtered_image2)
plt.show()



print('sss')