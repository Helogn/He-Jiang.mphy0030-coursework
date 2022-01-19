import numpy as np
import matplotlib.pyplot  as plt
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

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
        self.Coordinate_Matrix = Coordinate_Matrix


    def warp(self,affine):

        self.matrix = affine.trans
        
        self.Result2 = ndimage.affine_transform(self.Array,self.matrix,[0.5,0.5,2])

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
            Rotate_Matrix = Rotate_Matrix_x* Rotate_Matrix_y* Rotate_Matrix_z
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
            Rotate_Matrix = Rotate_Matrix_x* Rotate_Matrix_y* Rotate_Matrix_z
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
Transformation = Affinetransform([0,45,0,0,0,0,0])
result = obj.warp(Transformation)

plt.subplot(1, 2, 1)
plt.imshow(Image[20,:,:])
plt.title('Distance_transform_np')
plt.subplot(1, 2, 2)
plt.imshow(result[20,:,:])
plt.title('Distance_transform_edt')
plt.show()