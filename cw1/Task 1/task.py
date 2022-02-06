# HE JIANG
# 2021-12-2

# Task 1 Distance Transform in NumPy 

import numpy as np
import scipy.ndimage as sc
import time

def distance_transform_np( input_image , Space = [1,1,1] ):
    # takes a 3D volumetric binary image as input and returns its 3D Euclidean distance transform.
    # Input:  input_image    : binart image
    #         Space [z,y,x]  : voxel dimensions in each axis. default value [1,1,1]
    # Output: finial_output  : 3D Euclidean distance transform

    # algorithm: Calculation has two parts(loops):
    # Part one : Assume dimensions in each axis is 1 mm, indexing each non-zero voxel.
    #            All label voxel value is 1.
    #            Base on 6 nearest voxels, interior voxels will have bigger value than exterior voxels.
    #            If centre voxel equal to or smaller than 6-nearest voxels, the index of centre voxel will plus one.
    #            Loop all voxel use above method until all 'one' voxels have been calculated. Result will be like below:
    #            original:                      final:
    #            0 0 0 0 0 0                   0 0 0 0 0 0   
    #            0 1 1 1 0 0                   0 1 1 1 0 0   
    #            1 1 1 1 1 0                   1 2 3 2 1 0   
    #            0 1 1 1 1 0                   0 1 1 1 1 0   
    #            0 0 0 0 0 0                   0 0 0 0 0 0 
    # Part two : Due to impart of voxel dimensions(space) in each axis, use another loop 
    #            to calculate real 3D Euclidean distance transform.
    #            Treat each 'non-zero' voxel, its index almost means distance between itself and boundary.
    #            Create calculation kernal(n * n * n) for each voxel, length of kernal is half of its index.
    #            Distance in kernal is equal to in whole space due to above precalculation.
    #            One loop can calculate all 3D Euclidean distance

    input_dim = input_image.shape

    # define 6-nearest kernal
    kernal = np.zeros([3,3,3])
    kernal[0,1,1] = 1;kernal[1,0,1] = 1;kernal[1,2,1] = 1;kernal[1,1,0] = 1;kernal[1,1,2] = 1;kernal[2,1,1] = 1

    # expand output_image 2 voxels in each axis
    output_image = np.zeros([input_dim[0]+2,input_dim[1]+2,input_dim[2]+2])
    # finial output image
    finial_output = output_image.copy()
    # judge matrix
    judge = output_image.copy()
    judge[1:input_dim[0]+1,1:input_dim[1]+1,1:input_dim[2]+1] = input_image.copy()

    # to avoid influencing finial image, use a medium matrix to store medium values
    media_judge = judge.copy()

    # matrix of calculating Euclidean distance 
    cal_ary = judge.copy() 


    # Part One: calculate Index for non-zero voxel
    # design matrix
    SUM_of_nonzero_voxel = np.sum(judge)
    number = 1
    
    while  SUM_of_nonzero_voxel != 0 :
        
        for x in range(input_dim[0]):
            for y in range(input_dim[1]):
                for z in range(input_dim[2]):
                    
                    # judge if this voxel is zero or non-zero
                    if judge[x+1,y+1,z+1] != 0:  
                        
                        jug = (np.sum(judge[x:x+3,y:y+3,z:z+3] * kernal))

                        # sum < 6 means exist a nearest voxel which is equal to or bigger than centre voxel
                        if jug < 6:

                            output_image[x+1,y+1,z+1] = number
                            media_judge[x+1,y+1,z+1] = 0

        judge = media_judge.copy()
        
        SUM_of_nonzero_voxel = np.sum(judge)
        number = number + 1

    # calculate Euclidean distance
    for x in range(input_dim[0]):
        for y in range(input_dim[1]):
            for z in range(input_dim[2]):
                if cal_ary[x+1,y+1,z+1] == 1:
                    sz_of_kernal = int(output_image[x+1,y+1,z+1])

                    # according to index of centre voxel, create changing length of kernal.
                    cal_kernal = cal_ary[x+1-sz_of_kernal:x+2+sz_of_kernal,y+1-sz_of_kernal:y+2+sz_of_kernal,z+1-sz_of_kernal:z+2+sz_of_kernal]
                    result_kernal = cal_kernal.copy()

                    # calculate kernal distance
                    for i in range(sz_of_kernal*2 + 1):
                        for j in range(sz_of_kernal*2 + 1):
                            for k in range(sz_of_kernal*2 + 1):
                                if cal_kernal[i,j,k] == 0:
                                    result_kernal[i,j,k] = (((i - sz_of_kernal)*Space[0])**2 + ((j-sz_of_kernal)*Space[1])**2 + ((k-sz_of_kernal)*Space[2])**2) ** 0.5

                                else :
                                    result_kernal[i,j,k] = number

                    # select min value of multiplication which means minimum distance
                    finial_output[x+1,y+1,z+1] = np.min(result_kernal)
                    del result_kernal

    return finial_output[1:input_dim[0],1:input_dim[1],1:input_dim[2]]

# -----------------------------------------------------------------------------
# ----------------------------------main part ---------------------------------

# image path
Path_of_input = 'label_train00.npy'

# load image
data = np.load(Path_of_input)

# Compute Distance transform using distance_transform_np(np)
time1_start = time.time()
Result_of_np = distance_transform_np(data , [1,1,1] )
time1_end = time.time()

# Compute Distance transform using distance_transform_edt(edt)
time2_start = time.time()
Result_of_edt = sc.distance_transform_edt(data)
time2_end = time.time()

# print Running Time
# comment: distance_transform_edt is faster than my code.
#          according to traditional method in 2D calculation, PC can use only one loop to finish calculation.
#
#          However such a method is difficult to implement in a 3D image.
#          For my code, I used too many loops when I index all non-zero voxels, this part uses most part of my time.
#          If image is more complex, it would need more time to loop.
#
#          the key here is that looping wastes too much time.
print('\n\n-------------Running Time--------------------')
print("student_code running time = " + str(time1_end - time1_start))
print("edt running time =          " + str(time2_end - time2_start))
print('---------------------------------------------\n')

# compute mean and standard deviation of the voxel-level
# Comment: According to final images, my result is same as distance_transform_edt.
#          So it is clear my method is correct. 
#          And mean standard should be same between two method.
#          However according to type of data like double/float16, there might be few differences between them.

Mean_of_edt = np.mean(Result_of_edt)
std_of_edt = np.std(Result_of_edt)
Mean_of_np = np.mean(Result_of_np)
std_of_np = np.std(Result_of_np)
print('\nMean of edt: ' + str(round(Mean_of_edt,5)) + ' \nMean of np: ' + str(round(Mean_of_np,5)))
print('Std of edt: ' + str(round(std_of_edt,5)) + ' \nStd of np: ' + str(round(std_of_np,5)))

# Save 5 example slices to PNG files(Use matplotlib)

# plt.subplot(1, 2, 1)
# plt.imshow(Result_of_np[20,:,:])
# plt.title('Distance_transform_np')
# plt.subplot(1, 2, 2)
# plt.imshow(Result_of_edt[20,:,:])
# plt.title('Distance_transform_edt')
# plt.savefig('X_Slice=20')

# plt.subplot(1, 2, 1)
# plt.imshow(Result_of_np[:,46,:])
# plt.title('Distance_transform_np')
# plt.subplot(1, 2, 2)
# plt.imshow(Result_of_edt[:,46,:])
# plt.title('Distance_transform_edt')
# plt.savefig('Y_Slice=46')

# plt.subplot(1, 2, 1)
# plt.imshow(Result_of_np[:,40,:])
# plt.title('Distance_transform_np')
# plt.subplot(1, 2, 2)
# plt.imshow(Result_of_edt[:,40,:])
# plt.title('Distance_transform_edt')
# plt.savefig('Y_Slice=40')

# plt.subplot(1, 2, 1)
# plt.imshow(Result_of_np[:,:,40])
# plt.title('Distance_transform_np')
# plt.subplot(1, 2, 2)
# plt.imshow(Result_of_edt[:,:,40])
# plt.title('Distance_transform_edt')
# plt.savefig('Z_Slice=40')

# plt.subplot(1, 2, 1)
# plt.imshow(Result_of_np[:,:,45])
# plt.title('Distance_transform_np')
# plt.subplot(1, 2, 2)
# plt.imshow(Result_of_edt[:,:,45])
# plt.title('Distance_transform_edt')
# plt.savefig('Z_Slice=45')

# plt.show()