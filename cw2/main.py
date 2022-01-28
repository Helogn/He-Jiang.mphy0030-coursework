# 2022-1-2

from PIL import Image
import SimpleITK as sitk
from SimpleITK.SimpleITK import BilateralImageFilter
import matplotlib.pyplot  as plt
import matplotlib.image as img
import numpy as np
# from PIL import Image
import PIL


# -------------------reslice --------------------------- to line 158
# ------------------------------------------------------
# def reslice (Ori_Image,x,y,z = [0,0,1]):

#     sz = Ori_Image.shape


#     # ----------------for 2D image----------------------
#     if len(sz) == 2:

#         # mesh_x = (np.linspace(1, sz[0], sz[0],dtype = "int16" ) -1)
#         # mesh_y = (np.linspace(1, sz[1], sz[1],dtype = "int16" ) -1)
#         # # mesh_z = np.linspace(0,1,sz(2))
#         # xv, yv = np.meshgrid(mesh_x, mesh_y)
#         # Data_Matrix = np.array([mesh_x,mesh_x])

#         # Trans_matrix = [x,y]
#         # # Trans_matrix = np.transpose(Trans_matrix)
        
#         # result = Trans_matrix @ Data_Matrix


#         # --------------------------------------------
        
#         # Normalization
#         x1 = np.array(x)
#         y1 = np.array(y)
#         length_x = np.linalg.norm(x1)
#         length_y = np.linalg.norm(y1)

#         x = (x / length_x).tolist()
#         y = (y / length_y).tolist()

#         Coordinate_Matrix = []
        
#         for i in range (sz[0]):
#             for j in range (sz[1]):
#                 Coordinate_Matrix.append([i,j,Ori_Image[i,j]])

#         if len(x) == 2:
#             x.append(0)
#             y.append(0)

#         Trans_matrix = np.array([x,y,[0,0,1]])
#         Coordinate_Matrix = np.array(Coordinate_Matrix)
#         Coordinate_Matrix = np.transpose(Coordinate_Matrix)

#         # Transform matrix * Coordinate 
#         Result = np.dot(Trans_matrix, Coordinate_Matrix)
#         sz_Result = Result.shape
#         print("max Result" + str(np.max(Result,1)))
#         Range = np.int16(np.max(Result,1)-np.min(Result,1))

#         Result2 = np.floor(np.zeros([Range[0]+1,Range[1]+1])-1000)
#         print(Range)
#         for N in range (sz_Result[1]):
#             # print([Result[0,N],Result[1,N]])
#             tx = np.int16(Result[0,N])
#             ty = np.int16(Result[1,N])
#             Result2[tx,ty] = Result[2,N]
#             # print("point: "+str([Result[0,i],Result[1,i]]) + " value " + str(Result[2,i]))

#         # Result3 = Result2.tolist()
#         im = PIL.Image.fromarray(Result2)
#         # Test_im = im.resize( size = [sz[1],sz[0]], resample=2, box=None, reducing_gap=None)
#         sz1 = [sz[1],sz[0]]
#         Test_im = im.resize( size = sz1, resample=2, box=None, reducing_gap=None)
#         # Test_im.show()
#         Result4 = np.array(Test_im)

#         return Result4


#     if len(sz) == 3:

#         x1 = np.array(x)
#         y1 = np.array(y)
#         z1 = np.array(z)
#         length_x = np.linalg.norm(x1)
#         length_y = np.linalg.norm(y1)
#         length_z = np.linalg.norm(z1)

#         x = (x / length_x).tolist()
#         y = (y / length_y).tolist()
#         z = (z / length_z).tolist()

#         Coordinate_Matrix = []
        
#         for i in range (sz[0]):
#             for j in range (sz[1]):
#                 for k in range (sz[1]):
#                     Coordinate_Matrix.append([i,j,k,Ori_Image[i,j,k]])
#         print(' finish Creating Matrix')
#         if len(x) == 3:
#             x.append(0)
#             y.append(0)
#             z.append(0)

#         Trans_matrix = np.array([x,y,z,[0,0,0,1]])
#         Coordinate_Matrix = np.array(Coordinate_Matrix)
#         Coordinate_Matrix = np.transpose(Coordinate_Matrix)

#         Result = np.dot(Trans_matrix, Coordinate_Matrix)

#         print(' finish Calculating Matrix')
#         sz_Result = Result.shape
#         Range = np.int16(np.max(Result,1)-np.min(Result,1))

#         Result2 = np.floor(np.zeros([Range[0]+1,Range[1]+1,Range[2]+1])-1000)
#         print(sz_Result[1])
#         for N in range (sz_Result[1]):
#             # print([Result[0,N],Result[1,N]])
#             tx = np.int16(Result[0,N])
#             ty = np.int16(Result[1,N])
#             tz = np.int16(Result[2,N])
#             Result2[tx,ty,tz] = Result[3,N]
#             # print("point: "+str([Result[0,i],Result[1,i]]) + " value " + str(Result[2,i]))


# # -------------- calculate slowly ------------------------------
#         # X = np.linspace(min(Result[0,:]), max(Result[0,:]))
#         # Y = np.linspace(min(Result[1,:]), max(Result[1,:]))
#         # Z = np.linspace(min(Result[2,:]), max(Result[2,:]))
#         # print(' finish linspace')
#         # X, Y, Z = np.meshgrid(X, Y, Z)  # 2D grid for interpolation
        
#         # interp = LinearNDInterpolator(list(zip(Result[0,:],Result[1,:],Result[2,:])), Result[3,:])
#         # print(' finish Linear')
#         # V = interp(X, Y, Z)
#         # print(' finish interpolating')
# # ----------------------------------------------------------------
# # -------------- RBF ---------------------------------------------
# # rng = np.random.default_rng()


# # -------------------------------------------------------

#         return Result2


#     else :
#         return

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

def reslice (Ori_Image,x = [1,0,0],y = [0,1,0],z = [0,0,1]):
    size_of_image = Ori_Image.shape

    # Three Matrix to store transfered indexs of coordinate
    Coor_Matrix = np.zeros([3,size_of_image[0],size_of_image[1],size_of_image[2]])

    norm_X = np.linalg.norm(x)
    norm_Y = np.linalg.norm(y)
    norm_Z = np.linalg.norm(z)
    print('finish one ')
    for i in range(size_of_image[0]):
        for j in range(size_of_image[1]):
            for k in range(size_of_image[2]):
                Coor_Matrix[0,i,j,k] = np.dot([i,j,k],x)/norm_X
                Coor_Matrix[1,i,j,k] = np.dot([i,j,k],y)/norm_Y
                Coor_Matrix[2,i,j,k] = np.dot([i,j,k],z)/norm_Z

    print('finish two ')
    # create output image
    range_of_coor = np.zeros([3,2])
    range_of_coor[0,0] = np.max(Coor_Matrix[0,:,:,:]);range_of_coor[0,1] = np.min(Coor_Matrix[0,:,:,:])
    range_of_coor[1,0] = np.max(Coor_Matrix[1,:,:,:]);range_of_coor[1,1] = np.min(Coor_Matrix[1,:,:,:])
    range_of_coor[2,0] = np.max(Coor_Matrix[2,:,:,:]);range_of_coor[2,1] = np.min(Coor_Matrix[2,:,:,:])

    return norm_X

def nonlinear_filter(Image,iteration = 5, K = 30, L = 0.2):


    # Image : input Image
    # iteration: Times of Iteration
    
    # Ori_Img = image

    # halt
    sz = Image.shape
    DIM = len(sz)
    if DIM == 2:
        print(" DIM = 2 \n size = " + str(sz))
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
def resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkNearestNeighbor):
    """
    用itk方法将原始图像resample到与目标图像一致
    :param ori_img: 原始需要对齐的itk图像
    :param target_img: 要对齐的目标itk图像
    :param resamplemethod: itk插值方法: sitk.sitkLinear-线性  sitk.sitkNearestNeighbor-最近邻
    :return:img_res_itk: 重采样好的itk图像
    使用示范：
    import SimpleITK as sitk
    target_img = sitk.ReadImage(target_img_file)
    ori_img = sitk.ReadImage(ori_img_file)
    img_r = resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkLinear)
    """
    target_Size = target_img.GetSize()      # 目标图像大小  [x,y,z]
    target_Spacing = target_img.GetSpacing()   # 目标的体素块尺寸    [x,y,z]
    target_origin = target_img.GetOrigin()      # 目标的起点 [x,y,z]
    target_direction = target_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]

    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(target_Size)		# 目标图像大小
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    # 根据需要重采样图像的情况设置不同的dype
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt8)   # 近邻插值用于mask的，保存uint8
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))    
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
    return itk_img_resampled

# 0001.nii is image file
Path_of_image = './0001_image.nii'
Path_of_label = './0001_mask.nii'

# Read Image Data
data = sitk.ReadImage(Path_of_image )
array_of_data = sitk.GetArrayFromImage(data)
Slice = 100
# Reslice Part
# Resliced_image = reslice(array_of_data[Slice,:,:],x = [0,2], y = [2,0])
Resliced_image = reslice(array_of_data,[1,0,0],[0,1,0],[0,0,1])

# filtered5 = nonlinear_filter(Resliced_image,5,20,0.2)
# filtered10 = nonlinear_filter(Resliced_image,20,20,0.2)


# --------------------- plt reslice ----------------------

# plt.subplot(1, 2, 1)
# plt.imshow(array_of_data[Slice,:,:])
# plt.subplot(1, 2, 2)
# plt.imshow(Resliced_image[Slice,:,:])
# plt.show()
# ---------------------- plt nonlinear ---------------
# plt.subplot(1, 2, 1)
# plt.imshow(array_of_data[Slice,:,:])
# plt.subplot(1, 2, 2)
# plt.imshow(filtered10[Slice,:,:])
# plt.show()

plt.subplot(1, 2, 1)
plt.imshow(array_of_data[Slice,:,:])
plt.xlabel('X')
plt.ylabel('Y')
print(Resliced_image.shape)
plt.subplot(1, 2, 2)
plt.imshow(Resliced_image)
plt.show()






# --------------------------------------------------
# plt.subplot(2, 3, 1)
# plt.imshow(array_of_data[:,Slice,:],cmap='gray')
# plt.title('Original image')
# plt.subplot(2, 3, 2)
# # plt.imshow(np.squeeze(array_of_label[1,Slice,:,:]))
# plt.imshow(filtered5[:,Slice,:],cmap='gray')

# plt.subplot(2,3,3)
# plt.imshow(filtered10[:,Slice,:],cmap='gray')
# plt.subplot(2,3,5)
# plt.imshow(array_of_data[:,Slice,:] - filtered5[:,Slice,:],cmap='gray')
# plt.subplot(2,3,6)
# plt.imshow(array_of_data[:,Slice,:] - filtered10[:,Slice,:],cmap='gray')
# plt.show()



# data_label = sitk.ReadImage(Path_of_label )
# array_of_label = sitk.GetArrayFromImage(data_label)
# Result = reslice(array_of_data[Slice,:,:],[1,0],[0.1,1.1])



# ---------------------random
# from numpy import random
# sz = array_of_data.shape
# # Image_noise = array_of_data + random.standard_normal(sz) * 30

# filtered = nonlinear_filter(array_of_data,5,20,0.2)
# # ---------------------
# plt.subplot(2, 2, 1)
# # plt.imshow(np.squeeze(array_of_label[1,Slice,:,:]))
# plt.imshow(np.squeeze(array_of_label[1,Slice,:,:]))
# plt.title('Mask')

# plt.subplot(2, 2, 1)
# plt.imshow(array_of_data[Slice,:,:])
# plt.title('Original image')
# plt.subplot(2, 2, 2)
# # plt.imshow(Image_noise[Slice,:,:])
# plt.subplot(2, 2, 3)
# # plt.imshow(np.squeeze(array_of_label[1,Slice,:,:]))
# plt.imshow(filtered[Slice,:,:])

# plt.subplot(2,2,4)
# plt.imshow(array_of_data[Slice,:,:] - filtered[Slice,:,:])
# plt.show()



# ---------------------test----------------------------
# PNG_Path = 'R.jpg'
# IM = img.imread(PNG_Path)
# IM2 = nonlinear_filter(np.squeeze(IM[:,:,1]),10,1000,0.2)

# # plt.subplot(2,2,1)
# # plt.imshow(IM[:,:,1])
# # plt.title('Original Image')
# # plt.subplot(2,2,2)
# # plt.imshow(IM2)
# # plt.title('Filterd Image')
# # plt.subplot(2,2,3)
# # plt.imshow(np.squeeze(IM[:,:,1])-IM2)
# # plt.title('Difference')
# # plt.show()
# bin = np.linspace(-50,300,20)
# A,B = np.histogram(IM,bin)
# # print('hhh')
# plt.hist(A)
# plt.show()
# # plt.subplot(1,2,1)
# # plt.hist(IM[:,:,1],bin  )
# # plt.title("Original")
# # plt.subplot(1,2,2)
# # plt.hist(IM2,bin )
# # plt.title("Filtered")
# # plt.show()
# ---------------------test----------------------------


# IM = PIL.Image.open(PNG_Path)
# IM.show()

# Image.RASTERIZE