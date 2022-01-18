import numpy as np

class Image3D :

    def __init__(self,Array,dims =[1,1,1] ):


        self.Array = Array
        self.dims = dims

    def warp(self,parameter = []):

        

        self.d


class Affinetransform :
    def __init__(self, trans = None):
        
        length = len(self.trans)

        # judge length of vector
        if (length in [0,6,7,12]) == False:
            print('Error Input Vertor Length')
            return
        
        Matrix = np.zeros([4,4])
        # Rotate part


        if length == 6:
            # [Xangle,Yangle,Zangle,X_T,Y_T,Z_T]
            x = trans[0]
            y = trans[1]
            z = trans[2]
            R_X = [[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]]
            R_Y = [[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]]
            R_Z = [[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]]
            A3D = np.dot(R_X,np.dot(R_Y,R_Z))
            Matrix[0:3,0:3] = A3D

            # Translation Part
            Matrix[0,3] = trans[3]
            Matrix[1,3] = trans[4]
            Matrix[2,3] = trans[5]

            Matrix[3,3] = 1
            self.trans = Matrix

        if length == 7:
            # [Xangle,Yangle,Zangle,X_T,Y_T,Z_T,Scale]

            x = trans[0]
            y = trans[1]
            z = trans[2]
            R_X = [[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]]
            R_Y = [[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]]
            R_Z = [[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]]
            A3D = np.dot(R_X,np.dot(R_Y,R_Z))
            Matrix[0:3,0:3] = A3D * trans[6]


        



        self.trans = trans
    def rigid_transform (self, parameter = None):
        
        # judge data type
        if isinstance(parameter,(list,float,np.ndarray)) == False:
            parameter = np.array(parameter)
            print('Error Input Vertor Type')
            return

        self.parameter = parameter
        self.parameter = np.reshape(self.parameter,-1)
        length = len(self.parameter)

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
            Tx = parameter[0]
            Ty = parameter[1]
            Tz = parameter[2]


            # Rotate part
            Rotate_Matrix_x =np.array([[1, 0, 0], [0, np.cos(Rx), -np.sin(Rx)], [0, np.sin(Rx), np.cos(Rx)]])
            Rotate_Matrix_y =np.array([[np.cos(Ry), 0, np.sin(Ry)], [0, 1, 0], [-np.sin(Rx), 0, np.cos(Rx)]])
            Rotate_Matrix_z =np.array([[np.cos(Rx), -np.sin(Rx), 0], [np.sin(Rx), np.cos(Rx), 0], [0, 0, 1]])
            Rotate_Matrix = Rotate_Matrix_x* Rotate_Matrix_y* Rotate_Matrix_z
            # A3D = np.dot(R_X,np.dot(R_Y,R_Z))
            Homogeneous[0:3,0:3] = Rotate_Matrix

            # Translation Part
            T = [Tx, Ty, Tz]
            Homogeneous[0:3,3] = T

            Homogeneous[3,3] = 1
            
            return Homogeneous



    def affine_transform (self, parameter = None):
        
        # judge data type
        if isinstance(parameter,(list,float,np.ndarray)) == False:
            parameter = np.array(parameter)
            print('Error Input Vertor Type')
            return

        self.parameter = parameter
        self.parameter = np.reshape(self.parameter,-1)
        # length = len(self.parameter)

        # # judge length of vector
        # if (length in [0,6,7,12]) == False:
        #     print('Error Input Vertor Length')
        #     return

        # Random affine generation
        if length == 0:
            return


        if length == 7:
            return
        if length == 12:
            return

