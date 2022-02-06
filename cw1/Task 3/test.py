from cmath import nan
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
from scipy import ndimage

Image = plt.imread('SKY.png')
T = np.array([[1,0,-500], [0,1,0] ,[0,0,1]])
T = np.linalg.inv(T)
sz = Image.shape
p =  np.array(sz) / 2*-1
R = ndimage.affine_transform(Image,T,offset=p)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(Image)
plt.subplot(1,2,2)
plt.imshow(R)
plt.show()