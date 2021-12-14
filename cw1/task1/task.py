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
from numpy.core.arrayprint import array_repr
import scipy.ndimage as sc

# def convolutional_matrix():
#     np.convolve(output_image)
#     kernal = []
#     if x,y,z

def distance_transform_np( input_image , Dis = [1,1,1] ):

    Aim = np.sum(input_image)
    print(" Aim = " + str(Aim))

    input_dim = input_image.shape

    # define kernal
    kernal = np.zeros([3,3,3])
    kernal[0,1,1] = 1;kernal[1,0,1] = 1;kernal[1,2,1] = 1;kernal[1,1,0] = 1;kernal[1,1,2] = 1;kernal[2,1,1] = 1

    # expand output_image 2 voxels by each coordinate 
    output_image = np.zeros([input_dim[0]+2,input_dim[1]+2,input_dim[2]+2])
    judge = np.zeros([input_dim[0]+2,input_dim[1]+2,input_dim[2]+2])
    judge[1:input_dim[0]+1,1:input_dim[1]+1,1:input_dim[2]+1] = input_image.copy()
    media_judge = judge.copy()

    
    # design matrix
    SUM_all = np.sum(judge)
    print(SUM_all)
    number = 1
    
    while  SUM_all != 0 :
        
        for x in range(input_dim[0]):
            for y in range(input_dim[1]):
                for z in range(input_dim[2]):
                    
                    if judge[x+1,y+1,z+1] != 0:  #change

                        jug = (np.sum(judge[x:x+3,y:y+3,z:z+3] * kernal))

                        if jug < 6:

                            output_image[1+x,1+y,1+z] = number
                            media_judge[x+1,y+1,z+1] = 0

        judge = media_judge.copy()
        
        SUM_all = np.sum(judge)
        number = number + 1

        print('number = ' + str(number))
        print('SUM_all = ' + str(SUM_all))

    return output_image[1:input_dim[0],1:input_dim[1],1:input_dim[2]]

def distance_transform_np2( input_image , Dis = [1,1,1] ):
    

    Aim = np.sum(input_image)
    print(" Aim = " + str(Aim))

    input_dim = input_image.shape

    # define kernal
    kernal = np.zeros([3,3,3])
    kernal[0,1,1] = 1;kernal[1,0,1] = 1;kernal[1,2,1] = 1;kernal[1,1,0] = 1;kernal[1,1,2] = 1;kernal[2,1,1] = 1
    kernal[1,0,0] = 2;kernal[0,1,0] = 2;kernal[2,1,0] = 2;kernal[1,2,0] = 2;kernal[0,0,1] = 2;kernal[2,0,1] = 2
    kernal[2,2,1] = 2;kernal[0,2,1] = 2;kernal[1,0,2] = 2;kernal[2,1,2] = 2;kernal[1,2,2] = 2;kernal[0,1,2] = 2
    kernal[0,0,0] = 3;kernal[2,0,0] = 3;kernal[2,2,0] = 3;kernal[0,2,0] = 3;kernal[0,0,2] = 3;kernal[2,0,2] = 3
    kernal[2,2,2] = 3;kernal[0,2,2] = 3;kernal[1,1,1] = 4


    # expand output_image 2 voxels by each coordinate 
    output_image = np.zeros([input_dim[0]+2,input_dim[1]+2,input_dim[2]+2])
    judge = np.zeros([input_dim[0]+2,input_dim[1]+2,input_dim[2]+2])
    judge[1:input_dim[0]+1,1:input_dim[1]+1,1:input_dim[2]+1] = input_image.copy()
    media_judge = judge.copy()

    x_ray = np.zeros([input_dim[0]+2,input_dim[1]+2,input_dim[2]+2])
    y_ray = np.zeros([input_dim[0]+2,input_dim[1]+2,input_dim[2]+2])
    z_ray = np.zeros([input_dim[0]+2,input_dim[1]+2,input_dim[2]+2])

    SUM_all = np.sum(judge)
    print(SUM_all)
    
    # design matrix
    
    while  SUM_all != 0 :
        
        for x in range(input_dim[0]):
            for y in range(input_dim[1]):
                for z in range(input_dim[2]):
                    
                    if judge[x+1,y+1,z+1] != 0:  #change

                        jug_position = judge[x:x+3,y:y+3,z:z+3]*kernal

                        # seek 0 around the central point
                        jug_min = np.min(jug_position)
                        
                        # this point is closed to background
                        if jug_min == 0:

                            # pos1 = np.where(jug_position == 0)
                            jug3 = kernal.copy()
                            jug3[jug_position != 0] = 4
                            min1 = np.min(jug3)
                            pos = np.where(jug3 == min1)
                        
                            #-------------------------------------------
                            length_of_arr = np.size(pos,axis = 1)
                            jug = np.zeros([3,length_of_arr])
                            jug_value = np.zeros([length_of_arr])

                            for i in range(length_of_arr):
                                # jug[0][i] = 1
                                jug[0][i] = abs(pos[0][i]-1) + x_ray[x + pos[0][i],y+pos[1][i],z + pos[2][i]]
                                jug[1][i] = abs(pos[1][i]-1) + y_ray[x + pos[0][i],y+pos[1][i],z + pos[2][i]]
                                jug[2][i] = abs(pos[2][i]-1) + z_ray[x + pos[0][i],y+pos[1][i],z + pos[2][i]]

                                jug_value[i] = (jug[0][i])**2 + (jug[1][i])**2 + (jug[2][i])**2

                            max_jug_value = np.min(jug_value)
                            position_max_jug_value = np.where(jug_value == max_jug_value)
                            x_ray[x+1,y+1,z+1] = jug[0][position_max_jug_value[0][0]]
                            y_ray[x+1,y+1,z+1] = jug[1][position_max_jug_value[0][0]]
                            z_ray[x+1,y+1,z+1] = jug[2][position_max_jug_value[0][0]]

                            output_image[1+x,1+y,1+z] = jug_value[position_max_jug_value[0][0]]**0.5
                            media_judge[1+x,1+y,1+z] = 0


                            #------------------------------------------




                            # # x,y,z_ray save position of each voxel
                            # x_ray[x+1,y+1,z+1] = abs(pos[0][0]-1) + x_ray[x + pos[0][0],y+pos[1][0],z + pos[2][0]]
                            # y_ray[x+1,y+1,z+1] = abs(pos[1][0]-1) + y_ray[x + pos[0][0],y+pos[1][0],z + pos[2][0]]
                            # z_ray[x+1,y+1,z+1] = abs(pos[2][0]-1) + z_ray[x + pos[0][0],y+pos[1][0],z + pos[2][0]]
                            # # print(x_ray[x+1,y+1,z+1])
                            
                            # # calculate distance
                            # x_near = (x_ray[x+1,y+1,z+1])**2
                            # y_near = (y_ray[x+1,y+1,z+1])**2
                            # z_near = (z_ray[x+1,y+1,z+1])**2

                            # # save position
                            # output_image[1+x,1+y,1+z] = (x_near + y_near+ z_near)**0.5
                            # media_judge[1+x,1+y,1+z] = 0
                            # # print((x_near + y_near+ z_near)**0.5)

        judge = media_judge.copy()
        
        SUM_all = np.sum(judge)



        print('SUM_all = ' + str(SUM_all))

    return output_image[1:input_dim[0],1:input_dim[1],1:input_dim[2]]



Path_of_input = 'label_train00.npy'
# Path_of_input = 'image_train00.npy'

data = np.load(Path_of_input)
dim = [20,20,20]
image = distance_transform_np2(data , dim )



data = np.load(Path_of_input)
good = sc.distance_transform_edt(data)

plt.subplot(2, 2, 1)
plt.imshow(image[:,46,:])
plt.subplot(2, 2, 2)
plt.imshow(good[:,46,:])
plt.subplot(2, 2, 3)
plt.imshow(image[:,36,:])
plt.subplot(2, 2, 4)
plt.imshow(good[:,36,:])
plt.show()