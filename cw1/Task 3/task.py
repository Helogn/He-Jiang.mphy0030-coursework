import numpy as np
import matplotlib.pyplot  as plt
from scipy.interpolate import griddata
from scipy import ndimage

class Image3D :

    def __init__(self,Array):
        # input: 
        #        Array: input image for transfering
        #    

        self.sz = Array.shape

        # translate transformation centre point from corner to centre point
        self.offset = np.array(self.sz) / 2 *-1 
        self.Array = Array


    #     return output
    def warp(self,affine):
        # computes a warped 3D image, with all voxel intensities interpolated by trilinear interpolation method
        # input: 
        #       affine: homogeneous matrix
        # output:
        #       output: image 
        self.matrix = np.linalg.inv(affine)
        Transfered_image = ndimage.affine_transform(self.Array,self.matrix)

        return Transfered_image

class Affinetransform :
    def __init__(self, trans = []):
        # input parameter of homogeneous matrix
        #       parameter = [] : execute random_transform_generator
        #       parameter = [Rz,Ry,Rx,Tz,Ty,Tx] : from Rotate angle to Translation
        #       parameter = [Rz,Ry,Rx,Tz,Ty,Tx,Scale] : Rotate angle | Translation | scale
        #       parameter = [Rz,Ry,Rx,Tz,Ty,Tx,Sz,Sy,Sx,hz,hy,hx] : Rotate angle | Translation | scale | shear
        
        length = len(trans)

        # error wrong type input
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

        # print('parameter of matrix = \n' + str(self.trans))


    def rigid_transform (self, parameter ):
        # input:  parameterL  accept 6/7/12 numbers of parameter for transformation

        length = len(parameter)

        # judge length of vector
        if (length in [6,7,12]) == False:
            print('Error Input Vertor Length')
            return

        # Random affine generation

        # Rigid transformation DOF = 6
        if length == 6:

            Homogeneous = np.zeros([4,4])
            # parameter = []

            Rx = parameter[2] * 2 * np.pi /360
            Ry = parameter[1] * 2 * np.pi /360
            Rz = parameter[0] * 2 * np.pi /360
            Tx = parameter[5]
            Ty = parameter[4]
            Tz = parameter[3]


            # Rotate part
            Rotate_Matrix_z =np.array([[1, 0, 0], [0, np.cos(Rx), -np.sin(Rx)], [0, np.sin(Rx), np.cos(Rx)]])
            Rotate_Matrix_y =np.array([[np.cos(Ry), 0, np.sin(Ry)], [0, 1, 0], [-np.sin(Ry), 0, np.cos(Ry)]])
            Rotate_Matrix_x =np.array([[np.cos(Rz), -np.sin(Rz), 0], [np.sin(Rz), np.cos(Rz), 0], [0, 0, 1]])
            Rotate_Matrix = np.dot(Rotate_Matrix_z,np.dot(Rotate_Matrix_y, Rotate_Matrix_x))
            # A3D = np.dot(R_X,np.dot(R_Y,R_Z))
            Homogeneous[0:3,0:3] = Rotate_Matrix

            # Translation Part
            T = [Tz, Ty, Tx]
            Homogeneous[0:3,3] = T

            Homogeneous[3,3] = 1
            
            return Homogeneous

        # Rigid transformation DOF = 7
        if length == 7:
            Homogeneous = np.zeros([4,4])
            # [Zangle,Yangle,Xangle,Z_T,Y_T,X_T,Scale]

            Rx = parameter[2] * 2 * np.pi /360
            Ry = parameter[1] * 2 * np.pi /360
            Rz = parameter[0] * 2 * np.pi /360
            Tx = parameter[5]
            Ty = parameter[4]
            Tz = parameter[3]
            Scale = parameter[6]


            # Rotate part
            Rotate_Matrix_x =np.array([[1, 0, 0], [0, np.cos(Rx), -np.sin(Rx)], [0, np.sin(Rx), np.cos(Rx)]])
            Rotate_Matrix_y =np.array([[np.cos(Ry), 0, np.sin(Ry)], [0, 1, 0], [-np.sin(Ry), 0, np.cos(Ry)]])
            Rotate_Matrix_z =np.array([[np.cos(Rz), -np.sin(Rz), 0], [np.sin(Rz), np.cos(Rz), 0], [0, 0, 1]])
            Rotate_Matrix = np.dot(Rotate_Matrix_x,np.dot(Rotate_Matrix_y, Rotate_Matrix_z))

            Homogeneous[0:3,0:3] = Rotate_Matrix * Scale

            # Translation Part
            T = [Tx, Ty, Tz]
            Homogeneous[0:3,3] = T 

            Homogeneous[3,3] = 1
            
            return Homogeneous

    def affine_transform (self, parameter):
        
        length = len(parameter)

        Homogeneous = np.zeros([4,4])

        if length == 12:

            # [Zangle,Yangle,Xangle,Z_T,Y_T,X_T,Z_Scale,Y_Scale,X_Scale,Z_Shear,Y_shear,X_Shear]
            # above sequence[rotate,translation,scale,shear]

            Rx = parameter[2] * 2 * np.pi /360
            Ry = parameter[1] * 2 * np.pi /360
            Rz = parameter[0] * 2 * np.pi /360
            Tx = parameter[5]
            Ty = parameter[4]
            Tz = parameter[3]
            Scale = np.zeros([3,3])
            Scale[0,0] = abs(parameter[8])
            Scale[1,1] = abs(parameter[7])
            Scale[2,2] = abs(parameter[6])

            # Shear Matrix
            Shear = np.zeros([3,3])
            Shear[0,0] = 1;Shear[1,1] = 1;Shear[2,2] = 1
            Shear[0,1] = parameter[9];Shear[0,2] = parameter[11]
            Shear[1,0] = parameter[10];Shear[0,2] = parameter[10]
            Shear[2,0] = parameter[11];Shear[0,1] = parameter[9]

            # Rotate part
            Rotate_Matrix_z =np.array([[1, 0, 0], [0, np.cos(Rx), -np.sin(Rx)], [0, np.sin(Rx), np.cos(Rx)]])
            Rotate_Matrix_y =np.array([[np.cos(Ry), 0, np.sin(Ry)], [0, 1, 0], [-np.sin(Ry), 0, np.cos(Ry)]])
            Rotate_Matrix_x =np.array([[np.cos(Rz), -np.sin(Rz), 0], [np.sin(Rz), np.cos(Rz), 0], [0, 0, 1]])
            Rotate_Matrix = np.dot(Rotate_Matrix_z,np.dot(Rotate_Matrix_y, Rotate_Matrix_x))

            Homogeneous[0:3,0:3] = np.dot(Rotate_Matrix,np.dot(Shear,Scale))

            # Translation Part
            T = [Tx, Ty, Tz]
            Homogeneous[0:3,3] = T 
            Homogeneous[3,3] = 1

            return Homogeneous

    def random_transform_generator (self,strength = 1):
        # input: strength is powerful as strength increases

        Matrix = np.zeros([4,4])
        n, m = 3, 4

        # SVD makes sure u is an affine transformation matrix
        H = np.random.rand(n, m)
        u, s, vh = np.linalg.svd(H, full_matrices=False)
        u = np.round(u,4)
        Matrix[0:3,0:3] = u

        # define Homo Matrix
        Matrix[0:3,0:3] = u 
        Matrix[0:3,3] = H[:,0] 
        Matrix[3,3] = 1

        # define strength parameter
        # change translation strength
        Matrix[0:3,3] = H[:,0] * (strength + 1)
        
        # add Shear influence
        Matrix[0,1:3] = Matrix[0,1:3] * strength
        Matrix[1,0] = Matrix[1,0] * strength
        Matrix[2,0:2] = Matrix[2,0:2] * strength

        Matrix[0,0] = abs(Matrix[0,0]) +2
        Matrix[1,1] = abs(Matrix[1,1]) +2 
        Matrix[2,2] = abs(Matrix[2,2]) +2 

        return Matrix


# --------------------------------------------- ------------------------------------
# ------------------------------------main part ------------------------------------

# load image
Image1 = np.load('image_train00.npy')
obj = Image3D(Image1)
# Transformation = Affinetransform([0,10,0,2,0,0,2])
Transformation = Affinetransform()


# Manually define 10 rigid and affine transformation
T1 = Affinetransform([0,0,30,0,0,0]) # rotate
T2 = Affinetransform([45,0,0,0,0,0]) # rotate
T3 = Affinetransform([0,0,0,0,3,30])  # translation
T4 = Affinetransform([0,0,0,20,3,3])  # translation
T5 = Affinetransform([0,0,0,0,0,0,1])  # scale
T6 = T3.trans* T1.trans
T7 = T5.trans* T3.trans *T1.trans
T8 = T4.trans* T2.trans
T9 = T5.trans* T4.trans * T2.trans
T10 = T5.trans* T4.trans* T3.trans* T2.trans* T1.trans

# Generate the warped images using above transformations

warped_image1 = obj.warp(T1.trans)
warped_image2 = obj.warp(T2.trans)
warped_image3 = obj.warp(T3.trans)
warped_image4 = obj.warp(T4.trans)
warped_image5 = obj.warp(T5.trans)
warped_image6 = obj.warp(T6)
warped_image7 = obj.warp(T7)
warped_image8 = obj.warp(T8)
warped_image9 = obj.warp(T9)
warped_image10 = obj.warp(T10)



# Generate 10 different randomly warped images and plot 5 image slices for each transformed image at different z depths
Random = Affinetransform()
result = obj.warp(Random.trans)

# Generate images with 5 different values for the strength parameter
Homogeneous_Matrix = Random.random_transform_generator(1)
result1 = obj.warp(Homogeneous_Matrix)
Homogeneous_Matrix = Random.random_transform_generator(2)
result2 = obj.warp(Homogeneous_Matrix)
Homogeneous_Matrix = Random.random_transform_generator(3)
result3 = obj.warp(Homogeneous_Matrix)
Homogeneous_Matrix = Random.random_transform_generator(4)
result4 = obj.warp(Homogeneous_Matrix)
Homogeneous_Matrix = Random.random_transform_generator(5)
result5 = obj.warp(Homogeneous_Matrix)



