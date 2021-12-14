# HE JIANG
# 2021-12-2
# ------------------------------------------------------------------
# ------------------------------------------------------------------

# Task 1 Distance Transform in NumPy 
# • Implement a function distance_transform_np, which takes a 3D volumetric binary image as input and 
# returnsits 3D Euclidean distance transform. The function should accept the second argument as the 
# voxel dimensions in each axis and the computed distance transform should be in the unit of 
# millimetre. You should use onlyNumPy for implementing this function. [6]

# • Briefly describe the algorithm you used in the function docstring.[3]
# • Compare the built-in function distance_transform_edt in scipy.ndimage with your implementation.
# Implement a task script “task.py”, under folder “task1”, performing the following:
# o Download the “label_train00.npy” file, and use numpy.load to load. [1]
# o Compute distance transform of the segmentation boundary using the two implementations, 
# i.e. distance_transform_np and distance_transform_edt.[4]
# o Time the speed of two implementations, and comment on the difference.[3]
# o Compute the mean and standard deviation of the voxel-level difference between the two 
# implementations, and comment on the difference.[3]
# o Save 5 example slices to PNG files (filename being slice index), across the volume, together 
# with their corresponding distance transform results for each of the two algorithms.[5]

# ------------------------------------------------------------------
# ------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot  as plt
import scipy.ndimage as sc

# def convolutional_matrix():
#     np.convolve(output_image)
#     kernal = []
#     if x,y,z

def distance_transform_np( input_image , Dis = [1,1,1] ):
    # print(input_image.shape)
    # print(np.size(input_image))
    Aim = np.sum(input_image)
    print(" Aim = " + str(Aim))
    input_dim = input_image.shape
    min = np.amin(input_image) 
    max = np.amax(input_image)


    # expand output_image 2 voxels by each coordinate 
    output_image = np.zeros([input_dim[0]+2,input_dim[1]+2,input_dim[2]+2])
    output_image[1:input_dim[0]+1,1:input_dim[1]+1,1:input_dim[2]+1] = input_image
    output_image1 = output_image
    output_image2 = output_image
    
    # design matrix
    Ori = 1
    while True :
        Jug1 = 1
        Jug2 = 1
        for x in range(input_dim[0]):
            for y in range(input_dim[1]):
                for z in range(input_dim[2]):
                    # output_image[x+1,y+1,z+1]
                    if output_image1[x+1,y+1,z+1] != 0:
                        jug1 = np.amin([output_image1[x, y+1, z+1],output_image1[x+1, y, z+1],output_image1[x+1,y+1,z]])
                        if output_image1[x+1,y+1,z+1] <= jug1:
                            output_image1[x+1,y+1,z+1] = output_image1[x+1,y+1,z+1] + 1
                            Ori = Ori + 1
                            Jug1 = 1 + Jug1


                    x1 = input_dim[0]+1 -x
                    y1 = input_dim[1]+1 -y
                    z1 = input_dim[2]+1 -z
                    if output_image2[x1, y1 ,z1] != 0:
                        jug2 = np.amin([output_image2[x1+1 , y1, z1],output_image2[x1, y1+1, z1],output_image2[x1, y1, z1+1]]),
                        if output_image2[x1, y1 ,z1] <= jug2:
                            output_image2[x1, y1 ,z1] = output_image2[x1, y1 ,z1] + 1
                            Jug2 = Jug2 + 1




        if ( Ori > Aim and Jug1 == 1 and Jug2 == 1 ):
            break
    
    # output_image1, output_image2
    output_image = np.minimum(output_image1,output_image2)

    final_output_image = output_image[1:input_dim[0],1:input_dim[1],1:input_dim[2]]

    return output_image






    print("the max value of image is " + str(max))
    print("the min value of image is " + str(min))

    



    
    return output_image  #returnsits 3D Euclidean distance transform
class unknown:
    def __init__(self) -> None:
        pass



Path_of_input = 'label_train00.npy'
# Path_of_input = 'image_train00.npy'

data = np.load(Path_of_input)
dim = [20,20,20]
image = distance_transform_np(data , dim )
# c = len(image)

# np.save('risk1.npy',image)
# plt.imshow(image[:,40,:])
# plt.show()

# print(c)

data = np.load(Path_of_input)
good = sc.distance_transform_edt(data)

plt.subplot(1, 2, 1)
plt.imshow(image[:,36,:])
plt.subplot(1, 2, 2)
plt.imshow(good[:,36,:])
plt.show()