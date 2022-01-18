import numpy as np

class Image3D :

    def __init__(self,Array,dims =[1,1,1] ):


        self.Array = Array
        self.dims = dims

    def warp(self,parameter = []):

        

        self.d


class Affinetransform :
    def __init__(self, trans = None):
        
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
            return


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

            Rx = parameter[0]
            Ry = parameter[1]
            Rz = parameter[2]
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

            Rx = parameter[0]
            Ry = parameter[1]
            Rz = parameter[2]
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

    def random_transform_generator ():

        Matrix = np.zeros([4,4])


        return Matrix