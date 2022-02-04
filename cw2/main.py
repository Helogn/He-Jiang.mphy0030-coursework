# 2022-1-2
from numpy.lib.function_base import meshgrid
from scipy.interpolate import griddata
from PIL import Image
import SimpleITK as sitk
import matplotlib.pyplot  as plt
import matplotlib.image as img
import numpy as np

def reslice (Ori_Image,P1 = [0,0,0],P2 = [0,1,0],P3 = [0,0,1]):
    # z y x
    sz = Ori_Image.shape
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
    plt3d = plt.figure().gca(projection='3d')
    plt3d.scatter(xx, yy, zz)
    # plt.xlabel('x')
    # plt.show()
    
    # 3d interp
    z,y,x = meshgrid(range(sz[0]),range(sz[1]),range(sz[2]))

    A = np.array([np.reshape(z,-1),np.reshape(y,-1),np.reshape(x,-1)])
    B = np.array([np.reshape(zz,-1),np.reshape(yy,-1),np.reshape(xx,-1)])
    A = np.transpose(A)
    C = np.array([np.reshape(Ori_Image,-1)])
    C = np.transpose(C)
    interp = griddata(A, C, B,method='linear')


    size_of_image = Ori_Image.shape

def nonlinear_filter(Image,iteration = 5, K = 30, L = 0.2):


    # Image : input Image
    # iteration: Times of Iteration
    
    # Ori_Img = image

    # halt
    sz = Image.shape
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

# plt.imshow(np.squeeze(array_of_data[100,:,:]))
# plt.show()



Slice = 100

# Reslice Part
# Resliced_image = reslice(array_of_data[Slice,:,:],x = [0,2], y = [2,0])
Resliced_image = reslice(array_of_data,[1,1,1],[0,0,0],[1,1.6,2])

# ------------------- 3D-filtering before re-slicing ------------------------------

# ------------------------------------------------
# 3d_linear filter 
Filtered_slice = nonlinear_filter(array_of_data)
# Do a 3d slice
Result1 = reslice(Filtered_slice)

# ------------------------------------------------
#  reslice
Resliced_slice = reslice(array_of_data,[1,0,0],[0,0,0],[0,0,1])
# 2d Linear filter
sz = Resliced_slice.shape
Result2 = np.zeros(sz)
for i in range (sz[0]):
    Result2[i,:,:] = nonlinear_filter(Resliced_slice[i,:,:])

