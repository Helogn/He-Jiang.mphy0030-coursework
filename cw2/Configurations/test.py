
import SimpleITK as sitk
import matplotlib.pyplot  as plt
import matplotlib.image as img
from scipy.interpolate import griddata
import numpy as np
# from PIL import Image
import PIL



def reslice (Ori_Image,x = [1,0,0],y = [0,1,0],z = [0,0,1]):
    x = x/np.linalg.norm(x)
    y = y/np.linalg.norm(y)
    z = z/np.linalg.norm(z)
    size_of_image = Ori_Image.shape

    # Three Matrix to store transfered indexs of coordinate
    Coor_Matrix = np.zeros([3,size_of_image[0],size_of_image[1],size_of_image[2]])

    norm_X = np.linalg.norm(x)
    norm_Y = np.linalg.norm(y)
    norm_Z = np.linalg.norm(z)
    print('finish one ')
#     for i in range(size_of_image[0]):
#         for j in range(size_of_image[1]):
#             for k in range(size_of_image[2]):
#                 Coor_Matrix[0,i,j,k] = np.dot([i,j,k],x)/norm_X
#                 Coor_Matrix[1,i,j,k] = np.dot([i,j,k],y)/norm_Y
#                 Coor_Matrix[2,i,j,k] = np.dot([i,j,k],z)/norm_Z

                
    # faster method to calculate reslice coordinate

    # caiculate a row then expend to one plane
    for i in range (2):
        for k in range(size_of_image[2]):

            Coor_Matrix[0,i,0,k] = np.dot([i,0,k],x)/norm_X
            Coor_Matrix[1,i,0,k] = np.dot([i,0,k],y)/norm_Y
            Coor_Matrix[2,i,0,k] = np.dot([i,0,k],z)/norm_Z

            Coor_Matrix[0,i,1,k] = np.dot([i,1,k],x)/norm_X
            Coor_Matrix[1,i,1,k] = np.dot([i,1,k],y)/norm_Y
            Coor_Matrix[2,i,1,k] = np.dot([i,1,k],z)/norm_Z

        G_X = Coor_Matrix[0,i,1,:] - Coor_Matrix[0,i,0,:]
        G_Y = Coor_Matrix[1,i,1,:] - Coor_Matrix[1,i,0,:]
        G_Z = Coor_Matrix[2,i,1,:] - Coor_Matrix[2,i,0,:]

        for j in range(2,size_of_image[1]):
            Coor_Matrix[0,i,j,:] = Coor_Matrix[0,i,0,:] + G_X * j
            Coor_Matrix[1,i,j,:] = Coor_Matrix[1,i,0,:] + G_Y * j
            Coor_Matrix[2,i,j,:] = Coor_Matrix[2,i,0,:] + G_Z * j
    
    # calculate a plane then expand to a volume
    G_X = Coor_Matrix[0,1,:,:] - Coor_Matrix[0,0,:,:]
    G_Y = Coor_Matrix[1,1,:,:] - Coor_Matrix[1,0,:,:]
    G_Z = Coor_Matrix[2,1,:,:] - Coor_Matrix[2,0,:,:]
    
    for i in range (2,size_of_image[0]):
        Coor_Matrix[0,i,:,:] = Coor_Matrix[0,0,:,:] + G_X * i
        Coor_Matrix[1,i,:,:] = Coor_Matrix[1,0,:,:] + G_Y * i
        Coor_Matrix[2,i,:,:] = Coor_Matrix[2,0,:,:] + G_Z * i

    
        
        
    

    print('finish two ')
    # create output image
    range_of_coor = np.zeros([3,2])
    Coor_Matrix = np.round(Coor_Matrix)
    range_of_coor[0,0] = np.max(Coor_Matrix[0,:,:,:]);range_of_coor[0,1] = np.min(Coor_Matrix[0,:,:,:])
    range_of_coor[1,0] = np.max(Coor_Matrix[1,:,:,:]);range_of_coor[1,1] = np.min(Coor_Matrix[1,:,:,:])
    range_of_coor[2,0] = np.max(Coor_Matrix[2,:,:,:]);range_of_coor[2,1] = np.min(Coor_Matrix[2,:,:,:])
    Range_Cor = []
    Range_Cor.append(np.int16(range_of_coor[0,0] - range_of_coor[0,1]))
    Range_Cor.append(np.int16(range_of_coor[1,0] - range_of_coor[1,1]))
    Range_Cor.append(np.int16(range_of_coor[2,0] - range_of_coor[2,1]))
    Range_Cor = np.array(Range_Cor)

    Coor_Matrix[0,:,:,:] = Coor_Matrix[0,:,:,:] - range_of_coor[0,1]
    Coor_Matrix[1,:,:,:] = Coor_Matrix[1,:,:,:] - range_of_coor[1,1]
    Coor_Matrix[2,:,:,:] = Coor_Matrix[2,:,:,:] - range_of_coor[2,1]

    # calculate output

    C_X = np.reshape(Coor_Matrix[0,:,:,:],-1)
    C_Y = np.reshape(Coor_Matrix[1,:,:,:],-1)
    C_Z = np.reshape(Coor_Matrix[2,:,:,:],-1)
    Value = np.reshape(Ori_Image,-1)
    

    output = np.zeros([Range_Cor[0]+1,Range_Cor[1]+1,Range_Cor[2]+1])
    y1 = np.linspace(0,Range_Cor[1],Range_Cor[1]+1)
    z1 = np.linspace(0,Range_Cor[2],Range_Cor[2]+1)
    mesh_y, mesh_z  = np.meshgrid(y1,z1,indexing='ij')

    for i in range(Range_Cor[0]):

        Index = np.array(np.where(C_X == i))
        Y = C_Y[Index]
        Z = C_Z[Index]
        point = np.transpose(np.squeeze(np.array([Y,Z])))
        V = np.transpose(Value[Index])
        interp = griddata(point, V ,(mesh_y,mesh_z) ,method='linear')
        output[i,:,:] = np.squeeze(np.array(interp))
        # print(i)

    



    return output

Path_of_image = './0001_image.nii'
Path_of_label = './0001_mask.nii'

# Read Image Data
data = sitk.ReadImage(Path_of_image )
array_of_data = sitk.GetArrayFromImage(data)
Slice = 100
# Reslice Part
# Resliced_image = reslice(array_of_data[Slice,:,:],x = [0,2], y = [2,0])
Resliced_image = reslice(array_of_data,[0,1,0],[1,1,0.5],[1,0,0.5])
np.save('Resliced_image', Resliced_image)


plt.subplot(1, 2, 1)
plt.imshow(array_of_data[Slice,:,:])
plt.xlabel('X')
plt.ylabel('Y')
print(Resliced_image.shape)
plt.subplot(1, 2, 2)
plt.imshow(Resliced_image[Slice,:,:])
plt.show()
