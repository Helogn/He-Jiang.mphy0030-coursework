from task import Image3D,Affinetransform
import numpy as np
# import matplotlib.pyplot  as plt
from scipy.interpolate import griddata
from scipy import ndimage


Image1 = np.load('image_train00.npy')
obj = Image3D(Image1)
# Transformation = Affinetransform([0,10,0,2,0,0,2])
Transformation = Affinetransform()


# Manually define 10 rigid and affine transformation
T1 = Affinetransform([0,0,30,0,0,0]) # rotate
T2 = Affinetransform([45,0,0,0,0,0]) # rotate
T3 = Affinetransform([0,0,0,0,3,30])  # translation
T4 = Affinetransform([0,0,0,20,3,3])  # translation
T5 = Affinetransform([0,0,0,0,0,0,1])  # scale
T6 = T3.trans* T1.trans
T7 = T5.trans* T3.trans *T1.trans
T8 = T4.trans* T2.trans
T9 = T5.trans* T4.trans * T2.trans
T10 = T5.trans* T4.trans* T3.trans* T2.trans* T1.trans

# Generate the warped images using these transformations
warped_image1 = obj.warp(T1.trans)
warped_image2 = obj.warp(T2.trans)
warped_image3 = obj.warp(T3.trans)
warped_image4 = obj.warp(T4.trans)
warped_image5 = obj.warp(T5.trans)
warped_image6 = obj.warp(T6)
warped_image7 = obj.warp(T7)
warped_image8 = obj.warp(T8)
warped_image9 = obj.warp(T9)
warped_image10 = obj.warp(T10)



# Generate 10 different randomly warped images and plot 5 image slices for each transformed image at different z depths
Random = Affinetransform()
result = obj.warp(Random.trans)

# Generate images with 5 different values for the strength parameter
Homogeneous_Matrix = Random.random_transform_generator(1)
result1 = obj.warp(Homogeneous_Matrix)
Homogeneous_Matrix = Random.random_transform_generator(2)
result2 = obj.warp(Homogeneous_Matrix)
Homogeneous_Matrix = Random.random_transform_generator(3)
result3 = obj.warp(Homogeneous_Matrix)
Homogeneous_Matrix = Random.random_transform_generator(4)
result4 = obj.warp(Homogeneous_Matrix)
Homogeneous_Matrix = Random.random_transform_generator(5)
result5 = obj.warp(Homogeneous_Matrix)

print('Finish')

# plot/save part
# result = warped_image10

# plt.subplot(2,3,1)
# plt.imshow(result[5,:,:])
# plt.title('Z = 5')
# plt.gray()
# plt.subplot(2,3,2)
# plt.imshow(result[10,:,:])
# plt.title('Z = 10')
# plt.gray()
# plt.subplot(2,3,3)
# plt.imshow(result[15,:,:])
# plt.title('Z = 15')
# plt.gray()
# plt.subplot(2,3,4)
# plt.imshow(result[20,:,:])
# plt.title('Z = 20')
# plt.gray()
# plt.subplot(2,3,5)
# plt.imshow(result[25,:,:])
# plt.title('Z = 25')
# plt.gray()
# # plt.show()
# plt.savefig('Manual_defined_image10.png')