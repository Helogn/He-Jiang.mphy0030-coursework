# 2022-1-2
# from pathlib import Path
import SimpleITK as sitk
# from skimage.measure import marching_cubes
import matplotlib.pyplot  as plt
import numpy as np

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

    if len(sz) == 23:

        i = x
        j = y
        k = z


    Output_Image = []

    return Result2
# 0001.nii is image file
Path_of_image = './0001.nii'

data = sitk.ReadImage(Path_of_image )
array_of_data = sitk.GetArrayFromImage(data)


Result = reslice(array_of_data[50,:,:],[-1.5,1.5],[0,1])

plt.subplot(1, 2, 1)
plt.imshow(Result)
plt.subplot(1, 2, 2)
plt.imshow(array_of_data[50,:,:])
plt.title('Original image')
plt.show()
print(Result.shape)