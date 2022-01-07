# 2022-1-2
# from pathlib import Path
from PIL import Image
import SimpleITK as sitk
from SimpleITK.SimpleITK import BilateralImageFilter
# from skimage.measure import marching_cubes
import matplotlib.pyplot  as plt
import numpy as np
# from PIL import Image
import PIL

def reslice (Image,x = [1,0,0],y = [0,1,0],z = [0,0,1]):

    sz = Image.shape


    # ----------------for 2D image----------------------
    if len(sz) == 2:

        # mesh_x = (np.linspace(1, sz[0], sz[0],dtype = "int16" ) -1)
        # mesh_y = (np.linspace(1, sz[1], sz[1],dtype = "int16" ) -1)
        # # mesh_z = np.linspace(0,1,sz(2))
        # xv, yv = np.meshgrid(mesh_x, mesh_y)
        # Data_Matrix = np.array([mesh_x,mesh_x])

        # Trans_matrix = [x,y]
        # # Trans_matrix = np.transpose(Trans_matrix)
        
        # result = Trans_matrix @ Data_Matrix


        # --------------------------------------------
        
        # Normalization
        x1 = np.array(x)
        y1 = np.array(y)
        length_x = np.linalg.norm(x1)
        length_y = np.linalg.norm(y1)

        x = (x / length_x).tolist()
        y = (y / length_y).tolist()

        Coordinate_Matrix = []
        
        for i in range (sz[0]):
            for j in range (sz[1]):
                Coordinate_Matrix.append([i,j,Image[i,j]])

        if len(x) == 2:
            x.append(0)
            y.append(0)

        Trans_matrix = np.array([x,y,[0,0,1]])
        Coordinate_Matrix = np.array(Coordinate_Matrix)
        Coordinate_Matrix = np.transpose(Coordinate_Matrix)

        Result = np.dot(Trans_matrix, Coordinate_Matrix)
        sz_Result = Result.shape
        Range = np.int16(np.max(Result,1)-np.min(Result,1))

        Result2 = np.floor(np.zeros([Range[0]+1,Range[1]+1])-1000)
        print(sz_Result[1])
        for N in range (sz_Result[1]):
            # print([Result[0,N],Result[1,N]])
            tx = np.int16(Result[0,N])
            ty = np.int16(Result[1,N])
            Result2[tx,ty] = Result[2,N]
            # print("point: "+str([Result[0,i],Result[1,i]]) + " value " + str(Result[2,i]))

        # Result3 = Result2.tolist()
        im = PIL.Image.fromarray(Result2)
        # Test_im = im.resize( size = [sz[1],sz[0]], resample=2, box=None, reducing_gap=None)
        sz1 = [sz[1],sz[0]]
        Test_im = im.resize( size = sz1, resample=2, box=None, reducing_gap=None)
        # Test_im.show()
        Result4 = np.array(Test_im)


    if len(sz) == 23:

        i = x
        j = y
        k = z


    Output_Image = []

    return Result4


def nonlinear_filter(Image,iteration = 5, K = 0.3):

    # Image : input Image
    # iteration: Times of Iteration
    
    # Ori_Img = image

    # halt
    sz = Image.shape
    DIM = len(sz)
    if DIM == 2:
        
        # E_East =   np.array([[0,0,0],[0,-1,1],[0,0,0]])
        # E_West =   np.array([[0,0,0],[-1,1,0],[0,0,0]])
        # E_South =  np.array([[0,0,0],[0,-1,0],[0,1,0]])
        # E_North =  np.array([[0,-1,0],[0,1,0],[0,0,0]])
        for t in range(iteration):
            for i in range(sz[0]-1):
                for j in range(sz[1]-1):
                    N_I = Image[i,j+1] - Image[i,j] # North direction Calculation
                    S_I = Image[i,j] - Image[i,j-1] # Sorth direction Calculation
                    E_I = Image[i+1,j] - Image[i,j] # East direction Calculation
                    W_I = Image[i,j] - Image[i-1,j] # West direction Calculation

                    # diffusion function
                    C_N_I = np.exp(-N_I*N_I/K/K) # Diffusion function of Northern? direction
                    C_S_I = np.exp(-S_I*S_I/K/K) # Diffusion function of Northern? direction
                    C_E_I = np.exp(-E_I*E_I/K/K) # Diffusion function of Northern? direction
                    C_W_I = np.exp(-W_I*W_I/K/K) # Diffusion function of Northern? direction

                    Image = Image + K*(C_N_I*N_I + C_S_I*S_I + C_E_I*E_I + C_W_I)


    elif DIM == 3:

        for t in range(iteration):
            for i in range(sz[0]-1):
                
                for j in range(sz[1]-1):
                    
                    for k in range(sz[2]-1):
                    
                    
                        N_I = Image[i,j+1,k] - Image[i,j,k] # North direction Calculation
                        S_I = Image[i,j,k] - Image[i,j-1,k] # Sorth direction Calculation
                        E_I = Image[i+1,j,k] - Image[i,j,k] # East direction Calculation
                        W_I = Image[i,j,k] - Image[i-1,j,k] # West direction Calculation
                        I_I = Image[i,j,k+1] - Image[i,j,k] # infront direction Calculation
                        B_I = Image[i,j,k] - Image[i,j,k-1] # behind direction Calculation

                       # diffusion function
                        C_N_I = np.exp(-N_I*N_I/K/K) # Diffusion function of Northern? direction
                        C_S_I = np.exp(-S_I*S_I/K/K) # Diffusion function of Northern? direction
                        C_E_I = np.exp(-E_I*E_I/K/K) # Diffusion function of Northern? direction
                        C_W_I = np.exp(-W_I*W_I/K/K) # Diffusion function of Northern? direction
                        C_I_I = np.exp(-I_I*I_I/K/K) # Diffusion function of Northern? direction
                        C_B_I = np.exp(-B_I*B_I/K/K) # Diffusion function of Northern? direction

                        Image = Image + K*(C_N_I*N_I + C_S_I*S_I + C_E_I*E_I + C_W_I*W_I + C_I_I*I_I + C_B_I*B_I)
                    
        
    else:
        print('Wrong dimension of image')
        return


    return Image







# 0001.nii is image file
Path_of_image = './0001_image.nii'
Path_of_label = './0001_mask.nii'

data = sitk.ReadImage(Path_of_image )
array_of_data = sitk.GetArrayFromImage(data)
data_label = sitk.ReadImage(Path_of_label )
array_of_label = sitk.GetArrayFromImage(data_label)
print('shape of ' + str(array_of_label.shape))
Slice = 150
Result = reslice(array_of_data[Slice,:,:],[1,0],[0.1,1.1])

hhh = nonlinear_filter(array_of_data,1,0.2)

# im = Image.new( mode = "RGB", size = (200, 200), color = (153, 153, 255))
# im = Image.frombytes(mode = "RGB", size = (200, 200), data = Result ,decoder_name="raw")



# print(np.squeeze(array_of_label[1,:,Slice,:]).shape)
plt.subplot(2, 2, 1)
plt.imshow(np.squeeze(array_of_label[1,Slice,:,:]))
plt.title('Mask')
plt.subplot(2, 2, 2)
plt.imshow(array_of_data[Slice,:,:])
plt.title('Original image')
plt.subplot(2, 2, 3)
plt.imshow(hhh[10,:,:])

plt.show()
print(Result.shape)

# Image.RASTERIZE