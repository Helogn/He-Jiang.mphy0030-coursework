# 2022-1-2
from numpy.lib.function_base import meshgrid
# from PIL import Image
import SimpleITK as sitk
import matplotlib.pyplot  as plt
import matplotlib.image as img
import numpy as np

def interpolation_trilinear(image,point):
    # This function is to obtain interpolated 2D-plane from provided points' coordinates.
    # Use trilinear interpolation by utilising selected three points which are considered in the expected plane.
    # Return interpolated value from image.
    
    # Input:
    #       Image : 3D-array. 3D data which is used to obtain values of interpolated points.
    #       Point : array[3,n]. provided points which need to be interpolated.
    #               Column 0 is coordinate for the first point. this array includes n points' coordinates.
    # Output:
    #       output: 2D-array. interpolated plane points from above provided points

    

    point = np.array(point)
    sz = image.shape
    sz_point = point.shape
    output = np.zeros([1,sz_point[1]])
    for n in range(sz_point[1]):

        # calculate related 8 point value from P1 to P8
        z = point[0,n]
        y = point[1,n]
        x = point[2,n]

        floor_z = np.floor(z)
        ceil_z = np.floor(z+1)
        floor_y = np.floor(y)
        ceil_y = np.floor(y+1)
        floor_x = np.floor(x)
        ceil_x = np.floor(x+1)

        if floor_z >= 0 and ceil_z < sz[0] and floor_y >= 0 and ceil_y < sz[1] and floor_x >= 0 and ceil_x < sz[2]:
            
            P1 = tuple(np.int16([floor_z,floor_y,floor_x]))
            P2 = tuple(np.int16([floor_z,floor_y,ceil_x]))
            P3 = tuple(np.int16([floor_z,ceil_y ,ceil_x]))
            P4 = tuple(np.int16([floor_z,ceil_y,floor_x]))
            P5 = tuple(np.int16([ceil_z,floor_y,floor_x]))
            P6 = tuple(np.int16([ceil_z,floor_y,ceil_x]))
            P7 = tuple(np.int16([ceil_z,ceil_y,ceil_x]))
            P8 = tuple(np.int16([ceil_z,ceil_y,floor_x]))

            # calculate value of this voxel
            # x axes
            # print(image[P1])
            V12 = image[P1] * (x-P1[2]) + image[P2] * (P2[2] - x)
            V43 = image[P4] * (x-P4[2]) + image[P3] * (P2[2] - x)
            V56 = image[P5] * (x-P5[2]) + image[P6] * (P6[2] - x)
            V87 = image[P8] * (x-P8[2]) + image[P7] * (P7[2] - x)
            # if V12 > 0 :
                # print(V12)

            # y axis
            V1234 = V12 * (y - P1[1]) + V43 * (P4[1] - y)
            V5678 = V56 * (y - P5[1]) + V87 * (P8[1] - y)

            # z axis
            output[0,n] = V1234* (z - P1[0]) + V5678 * (P5[0] - z)
            # print(output[0,n])

    return np.array(output)


def reslice (Ori_Image,P1 = [0,0,0],P2 = [0,1,0],P3 = [0,0,1]):
    # This function is to obtain a slice from input 3D data. 
    # User should select three points which are considered in the expected plane.
    # This function will calculate related plane function and interpolate related points.
    
    # Input:
    #       Image : array/list.  Input 3D data.
    #       P1,P2,P3: array[Z,Y,X]. includes coordinates in three axises(Z,Y,X) 
    # Output:
    #       zz, yy, xx: array. meshgrid in three axises. each size is same as input data.
    #       output: array[Y,X]. resliced plane in 2D. This 2d array can also be regard as z value for given x,y variants. 


    sz = Ori_Image.shape

    Ori_max = np.max(Ori_Image)
    Ori_min = np.min(Ori_Image)
    # Ori_Image = (Ori_Image - Ori_min)/(Ori_max - Ori_min) * (255*2) - 255
   # A = y1 (z2 - z3) + y2 (z3 - z1) + y3 (z1 - z2)
    A = P1[1]*(P2[0] - P3[0]) + P2[1]*(P3[0] - P1[0]) + P3[1]*(P1[0] - P2[0])
  #  B = z1 (x2 - x3) + z2 (x3 - x1) + z3 (x1 - x2)
    B = P1[0]*(P2[2] - P3[2]) + P2[0]*(P3[2] - P1[2]) + P3[0]*(P1[2] - P2[2])
  # C = x1 (y2 - y3) + x2 (y3 - y1) + x3 (y1 - y2)
    C = P1[2]*(P2[1] - P3[1]) + P2[2]*(P3[1] - P1[1]) + P3[2]*(P1[1] - P2[1])
  #  D = x1 (y2 z3 - y3 z2) + x2 (y3 z1 - y1 z3) + x3 (y1 z2 - y2 z1)
    D = P1[2]*(P2[1]*P3[0] - P3[1]*P2[0]) + P2[2]*(P3[1]*P1[0] - P1[1]*P3[0]) + P3[2]*(P1[1]*P2[0] - P2[1]*P1[0])
    # normalise input vector

    xx, yy = np.meshgrid(range(sz[2]), range(sz[1]))
    
    zz = (-A * xx - B * yy - D) * 1. /C

    # plt.xlabel('x')
    # plt.show()

    x = np.reshape(xx,-1)
    y = np.reshape(yy,-1)
    z = np.reshape(zz,-1)

    output = interpolation_trilinear(Ori_Image,[z,y,x])

    output = np.reshape(output,[sz[1],sz[2]])
    return zz, yy, xx, output


def nonlinear_filter(Image,iteration = 5, K = 30, L = 0.2):
    # This function is to do a nonlinear filter for input 3D data and return a 3D data with same size
    # Perona and Malik developed a smoothing and edge detection method. 
    # Their anisotropic diffusion filtering method is mathematically formulated as a diffusion process

    # Input:
    #       Image : 3D-array/list.  Input 3D data.
    #       iteration: constant. Times of iteration.
    #       K:  constant. control depth of filter. bigger K means powerful filter.
    #       L:  constant. parameter of equation. between 0.2-0.25.
    # Output:
    #       O_Image: 3D-array. size is same as input data
    

    Image = np.array(Image)
    sz = Image.shape

    # judge 2D or 3D data
    DIM = len(sz)
    if DIM == 2:
        # print(" DIM = 2 \n size = " + str(sz))
        # --------------------------------------------------------------
        O_Image = np.zeros([sz[0]+2,sz[1]+2])

        for t in range(iteration):
            N_Image = O_Image.copy()
            S_Image = O_Image.copy()
            E_Image = O_Image.copy()
            W_Image = O_Image.copy()

            O_Image[1:sz[0]+1,1:sz[1]+1] = Image
            N_Image[1:sz[0]+1,2:sz[1]+2] = Image
            S_Image[1:sz[0]+1,0:sz[1]  ] = Image
            E_Image[2:sz[0]+2,1:sz[1]+1] = Image
            W_Image[0:sz[0]  ,1:sz[1]+1] = Image

            N_Image = N_Image - O_Image; C_N_Image = np.exp(-N_Image*N_Image/K/K)
            S_Image = O_Image - S_Image; C_S_Image = np.exp(-S_Image*S_Image/K/K)
            E_Image = E_Image - O_Image; C_E_Image = np.exp(-E_Image*E_Image/K/K)
            W_Image = O_Image - W_Image; C_W_Image = np.exp(-W_Image*W_Image/K/K)
            
            # print(A)
            # print("hhhh")
            O_Image = O_Image + L*(N_Image*C_N_Image + S_Image*C_S_Image + E_Image*C_E_Image+ W_Image*C_W_Image )

        return O_Image[1:sz[0]+1,1:sz[1]+1]

    elif DIM == 3:
        
        O_Image = np.zeros([sz[0]+2,sz[1]+2,sz[2]+2])
        for t in range(iteration):

            N_Image = O_Image.copy()
            S_Image = O_Image.copy()
            E_Image = O_Image.copy()
            W_Image = O_Image.copy()
            I_Image = O_Image.copy()
            B_Image = O_Image.copy()

            O_Image[1:sz[0]+1,1:sz[1]+1,1:sz[2]+1] = Image
            N_Image[1:sz[0]+1,2:sz[1]+2,1:sz[2]+1] = Image
            S_Image[1:sz[0]+1,0:sz[1]  ,1:sz[2]+1] = Image
            E_Image[2:sz[0]+2,1:sz[1]+1,1:sz[2]+1] = Image
            W_Image[0:sz[0]  ,1:sz[1]+1,1:sz[2]+1] = Image
            I_Image[1:sz[0]+1,1:sz[1]+1,2:sz[2]+2] = Image
            B_Image[1:sz[0]+1,1:sz[1]+1,0:sz[2]  ] = Image

            N_Image = N_Image - O_Image; C_N_Image = np.exp(-N_Image*N_Image/K/K)
            S_Image = O_Image - S_Image; C_S_Image = np.exp(-S_Image*S_Image/K/K)
            E_Image = E_Image - O_Image; C_E_Image = np.exp(-E_Image*E_Image/K/K)
            W_Image = O_Image - W_Image; C_W_Image = np.exp(-W_Image*W_Image/K/K)
            I_Image = I_Image - O_Image; C_I_Image = np.exp(-I_Image*I_Image/K/K)
            B_Image = O_Image - B_Image; C_B_Image = np.exp(-B_Image*B_Image/K/K)

            O_Image = O_Image + L*(N_Image*C_N_Image + S_Image*C_S_Image + E_Image*C_E_Image+ W_Image*C_W_Image + I_Image*C_I_Image + B_Image*C_B_Image)

        return O_Image[1:sz[0]+1,1:sz[1]+1,1:sz[2]+1]

    else:
        print('Wrong dimension of image')
        return

# Image_Path
Path_of_image = '0001_image.nii'
Path_of_label = './0001_mask.nii'

# Read Image Data
data = sitk.ReadImage(Path_of_image )
array_of_data = sitk.GetArrayFromImage(data)


# Analyse the impact due to varying filter parameter values
Filtered_image1 = nonlinear_filter(array_of_data,iteration = 3, K = 10, L = 0.2)

Filtered_image2 = nonlinear_filter(array_of_data,iteration = 1, K = 10, L = 0.2)

Filtered_image3 = nonlinear_filter(array_of_data,iteration = 3, K = 20, L = 0.2)


# ------------------- 3D-filter before re-slicing ------------------------------
# 3d_linear filter 
Filtered_slice = nonlinear_filter(array_of_data)
print('finish filter')

# reslice
z1, y1, x1, Result1 = reslice(Filtered_slice,[0,0,0],[50,150,0],[50,150,100])
print('finish reslice')

# ------------------- 3d-filter after re-slicing ------------------------------
# reslice
z2, y2, x2, Resliced_slice = reslice(array_of_data,[0,0,0],[50,150,0],[50,150,100])
print('finish reslice')
# 2d Linear filter
Result2 = nonlinear_filter(Resliced_slice)
print('finish filter')

print('mean of Result2   ' + str(np.mean(Result1)))
print('mean of Result1   ' + str(np.mean(Result2)))
print('std of Result2   ' + str(np.std(Result1)))
print('std of Result1   ' + str(np.std(Result2)))
print('mean of Subtraction between Result1 and Result2   ' + str(np.mean(Result1 - Result2)))
print('std of Subtraction  between Result1 and Result2   ' + str(np.std(Result1 - Result2)))


# -------------------- utilise the organ segmentation to help comparison ----------------------
print('\n\n--------utilise the organ segmentation to help comparison---------')

Label = sitk.ReadImage(Path_of_label)
array_of_label = sitk.GetArrayFromImage(Label)
array_of_data[array_of_label != 1] == 0 




# -------------------------------------- plot part ----------------------------------------
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(Result1)
# plt.title('3D-filtering before re-slicing')
# plt.colorbar()
# plt.subplot(1,2,2)
# plt.imshow(Result2)
# plt.title('re-slicing after 3d-filter')
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(Resliced_slice)
# plt.title('resliced image in 2D')
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.gray()
# plt.show()



# # plt3d_1 = plt.figure().gca(projection='3d')
# # plt3d_1.scatter(np.reshape(z1,-1), np.reshape(y1,-1), np.reshape(x1,-1), c = np.reshape(Result1,-1),cmap = 'gray')

# plt3d_2 = plt.figure().gca(projection='3d')
# plt3d_2.scatter(np.reshape(z2,-1), np.reshape(y2,-1), np.reshape(x2,-1), c = np.reshape(Result2,-1),cmap = 'gray')
# plt.title('resliced image in 3D')
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.show()

