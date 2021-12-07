import numpy as np

# a = np.ones([2,2])
# print(a)
# c = np.zeros([4,4])
# print(c)
# c[1:3,1:3] = a
# print(1:3)
# b = [2,2]
# print(b)
# b = b +1
# print(b)


# a = [5 , 2 , 3,2,2,7,9]
# a2 = [2 , 3 , 5,7,9,2,2]
# a3 = [1 , 1 , 1,1,1,1,1]
# # a3[a2>a] = a2[a2<=a]
# c = np.minimum(a,a2)
# print(c)

import numpy as np
import matplotlib.pyplot  as plt
import scipy.ndimage as sc
Path_of_input = 'label_train00.npy'
Path_of_jiang = 'risk1.npy'
data = np.load(Path_of_input)
good = sc.distance_transform_edt(data)
data_j = np.load(Path_of_jiang)

plt.subplot(1, 2, 1)
plt.imshow(data_j[:,36,:])
plt.subplot(1, 2, 2)
plt.imshow(good[:,38,:])
plt.show()