import numpy as np

class Image3D :

    def __init__(self,Array,dims =[1,1,1] ):


        self.Array = Array
        self.dims = dims

    def warp(self,parameter = []):

        

        self.d


class Affinetransform :
    def __init__(self, trans):

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
            Homogeneous = np.zeros([4,4])

        
            T = [Tx, Ty, Tz]


    def affine_transform (self, parameter = None):
        
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


        if length == 7:
            return
        if length == 12:
            return

