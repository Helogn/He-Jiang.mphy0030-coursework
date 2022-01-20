import numpy as np
import matplotlib.pyplot  as plt
from scipy.interpolate import LinearNDInterpolator

class Image3D :

    def __init__(self,Array,dims =[1,1,1] ):


        self.Array = Array
        self.dims = dims
        
        sz = Array.shape
        Coordinate_Matrix = []
        
        for i in range (sz[0]):
            for j in range (sz[1]):
                for k in range (sz[1]):
                    Coordinate_Matrix.append([i,j,k,Array[i,j,k]])
        print(' finish Creating Matrix')
        Coordinate_Matrix = np.array(Coordinate_Matrix)
        Coordinate_Matrix = np.transpose(Coordinate_Matrix)
        self.Value_Matrix = Coordinate_Matrix.copy()
        Coordinate_Matrix[3,:] = 1
        print('max = ' + str(np.max(Coordinate_Matrix,1)))
        print('min = ' + str(np.min(Coordinate_Matrix,1)))
        self.Coordinate_Matrix = Coordinate_Matrix


    def warp(self,affine):

        self.matrix = affine.trans
        
        Result = np.dot(self.matrix, self.Coordinate_Matrix)
        # Value_result = np.transpose(Result)
        Result[3,:] = self.Value_Matrix[3,:]

        print(' finish Calculating Matrix')
        sz_Result = Result.shape
        Max = np.max(Result,1)
        Min = np.min(Result,1)
        # ------------------ +- ------------------------
        Result[0,:] = Result[0,:] - Min[0]
        Result[1,:] = Result[1,:] - Min[1]
        Result[2,:] = Result[2,:] - Min[2]

        Max = np.max(Result,1)
        Min = np.min(Result,1)
        Range = np.int16(Max)-np.int16(Min)
        # -------------------------------------------
        self.Result2 = np.floor(np.zeros([Range[0]+1,Range[1]+1,Range[2]+1],dtype='int32')-1000)
        print(sz_Result[1])
        for N in range (sz_Result[1]):

            tx = np.int16(np.floor(Result[0,N])-self.matrix[0,3])
            ty = np.int16(np.floor(Result[1,N])-self.matrix[1,3])
            tz = np.int16(np.floor(Result[2,N])-self.matrix[2,3])
            self.Result2[tx,ty,tz] = (Result[3,N])

        # print('finish Transfor')


        # interpolation        
        x = Result[0,:]
        y = Result[1,:]
        z = Result[2,:]
        xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
        my_interpolating_function = LinearNDInterpolator(np.transpose(Result[0:3,:]), np.transpose(Result[3,:]))
        # print(my_interpolating_function([1,1,1]))
        



        return self.Result2


class Affinetransform :
    def __init__(self, trans = []):
        
        length = len(trans)

        if isinstance(trans,(list,float,np.ndarray)) == False:
            trans = np.array(trans)
            print('Error Input Vertor Type')
            return

        self.trans = trans
        self.trans = np.reshape(self.trans,-1)
        length = len(self.trans)

        # judge length of vector
        if (length in [0,6,7,12]) == False:
            print('Error Input Vertor Length')
            return

        # Random affine generation
        if length == 0:
            self.trans = self.random_transform_generator()


        if length == 6:
            self.trans = self.rigid_transform(trans)

        if length == 7:
            self.trans = self.rigid_transform(trans)

        if length == 12:
            self.trans = self.affine_transform(trans)


    def rigid_transform (self, parameter = None):
        
        # judge data type

        length = len(parameter)

        # judge length of vector
        if (length in [0,6,7,12]) == False:
            print('Error Input Vertor Length')
            return

        # Random affine generation
        if length == 0:
            return

        # Rigid transformation DOF = 6
        if length == 6:
            Homogeneous = np.zeros([4,4])
            # [Xangle,Yangle,Zangle,X_T,Y_T,Z_T,Scale]

            Rx = parameter[0] * 2 * np.pi /360
            Ry = parameter[1] * 2 * np.pi /360
            Rz = parameter[2] * 2 * np.pi /360
            Tx = parameter[3]
            Ty = parameter[4]
            Tz = parameter[5]


            # Rotate part
            Rotate_Matrix_x =np.array([[1, 0, 0], [0, np.cos(Rx), -np.sin(Rx)], [0, np.sin(Rx), np.cos(Rx)]])
            Rotate_Matrix_y =np.array([[np.cos(Ry), 0, np.sin(Ry)], [0, 1, 0], [-np.sin(Ry), 0, np.cos(Ry)]])
            Rotate_Matrix_z =np.array([[np.cos(Rz), -np.sin(Rz), 0], [np.sin(Rz), np.cos(Rz), 0], [0, 0, 1]])
            Rotate_Matrix = np.dot(Rotate_Matrix_x,np.dot(Rotate_Matrix_y, Rotate_Matrix_z))
            # A3D = np.dot(R_X,np.dot(R_Y,R_Z))
            Homogeneous[0:3,0:3] = Rotate_Matrix

            # Translation Part
            T = [Tx, Ty, Tz]
            Homogeneous[0:3,3] = T

            Homogeneous[3,3] = 1
            
            return Homogeneous

        # Rigid transformation DOF = 7
        if length == 7:
            Homogeneous = np.zeros([4,4])
            # [Xangle,Yangle,Zangle,X_T,Y_T,Z_T,Scale]

            Rx = parameter[0] * 2 * np.pi /360
            Ry = parameter[1] * 2 * np.pi /360
            Rz = parameter[2] * 2 * np.pi /360
            Tx = parameter[3]
            Ty = parameter[4]
            Tz = parameter[5]
            Scale = parameter[6]


            # Rotate part
            Rotate_Matrix_x =np.array([[1, 0, 0], [0, np.cos(Rx), -np.sin(Rx)], [0, np.sin(Rx), np.cos(Rx)]])
            Rotate_Matrix_y =np.array([[np.cos(Ry), 0, np.sin(Ry)], [0, 1, 0], [-np.sin(Ry), 0, np.cos(Ry)]])
            Rotate_Matrix_z =np.array([[np.cos(Rz), -np.sin(Rz), 0], [np.sin(Rz), np.cos(Rz), 0], [0, 0, 1]])
            Rotate_Matrix = np.dot(Rotate_Matrix_x,np.dot(Rotate_Matrix_y, Rotate_Matrix_z))
            # A3D = np.dot(R_X,np.dot(R_Y,R_Z))
            Homogeneous[0:3,0:3] = Rotate_Matrix * Scale

            # Translation Part
            T = [Tx, Ty, Tz]
            Homogeneous[0:3,3] = T 

            Homogeneous[3,3] = 1
            
            return Homogeneous

    def affine_transform (self, par = None):
        
        # judge data type
        length = len(par)

        Homogeneous = np.zeros([4,4])

        if length == 12:

            Homogeneous[0:3,0:3] = [[par[0],par[1],par[2]],[par[3],par[4],par[5]],[par[6],par[7],par[8]]]
            Homogeneous[3,3] = 1

            return Homogeneous

    def random_transform_generator (self):

        Matrix = np.zeros([4,4])
        n, m = 3, 4

        H = np.random.rand(n, m)
        u, s, vh = np.linalg.svd(H, full_matrices=False)
        u = np.round(u,4)
        # Matrix = u @ vh
        Matrix[0:3,0:3] = u
        Matrix[3,3] = 1

        return Matrix


Image = np.load('image_train00.npy')
obj = Image3D(Image)
# Transformation = Affinetransform([30,0,0,2,0,0,2])
Transformation = Affinetransform()
result = obj.warp(Transformation)

print(result.shape)
plt.subplot(2, 2, 1)
plt.imshow(Image[10,:,:])

plt.subplot(2, 2, 2)
plt.imshow(result[50,:,:])
# plt.xlabel('x')
# plt.subplot(2, 2, 3)
# plt.imshow(Image[:,50,:])
# plt.subplot(2, 2, 4)
# plt.imshow(result[:,50,:])

plt.show()