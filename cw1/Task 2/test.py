import numpy as np

from scipy.ndimage import gaussian_filter
a = np.arange(2500, step=1).reshape((50,50))


from scipy import misc
import matplotlib.pyplot as plt
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(221)  # left side
ax2 = fig.add_subplot(222)  # right side
ax3 = fig.add_subplot(223)  # right side
ax4 = fig.add_subplot(224)  # right side
ascent = misc.ascent()
result = gaussian_filter(ascent, sigma=5)
ax1.imshow(ascent)
ax2.imshow(gaussian_filter(ascent, sigma=0.1))
ax3.imshow(gaussian_filter(ascent, sigma=0.5))
ax4.imshow(gaussian_filter(ascent, sigma=1.0))
plt.show()