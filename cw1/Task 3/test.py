# def mymax(a:str = "12345",b:str = "123456789"):
#     return max(len(a),len(b))
# def mymax(a :int = 0,b:int = 1):
#     return max(a,b)

# print(mymax("one","three"))
# print(mymax(1,3))#AllArgs
# print(mymax())#NoArgs
import numpy as np
class Affinetransform :
    def __init__(self, parameter):

        # judge data type
        if isinstance(parameter,(list,float,np.ndarray)) == False:
            parameter = np.array(parameter)
            print('Error Input Vertor Type')
            return
        self.parameter = parameter

        self.parameter = np.reshape(self.parameter,-1)
        self.length = len(self.parameter)

        # judge length of vector
        if (self.length in [0,6,7,12]) == False:
            print('Error Input Vertor Length')
            return


        Homogeneous_Rigid = np.zeros([4,4])
        Homogeneous_Rigid[3,3] = 1

        # Define Rigid Transformation DOF = 6
        if self.length == 6:
            Rx = parameter[0]
            Ry = parameter[1]
            Rz = parameter[2]
            Tx = parameter[3]
            Ty = parameter[4]
            Tz = parameter[5]


            # Rotate part
            Rotate_Matrix_x =np.array([[1, 0, 0], [0, np.cos(Rx), -np.sin(Rx)], [0, np.sin(Rx), np.cos(Rx)]])
            Rotate_Matrix_y =np.array([[np.cos(Ry), 0, np.sin(Ry)], [0, 1, 0], [-np.sin(Rx), 0, np.cos(Rx)]])
            Rotate_Matrix_z =np.array([[np.cos(Rz), -np.sin(Rz), 0], [np.sin(Rz), np.cos(Rz), 0], [0, 0, 1]])
            Rotate_Matrix = Rotate_Matrix_x* Rotate_Matrix_y* Rotate_Matrix_z

            # Translate part
            T = [Tx, Ty, Tz]

            # Homogeneous
            Homogeneous_Rigid[0:3,0:3] = Rotate_Matrix
            Homogeneous_Rigid[0:3,3] = T


        # Define Rigid Transformation DOF = 7
        if self.length == 7:

            Rx = parameter[0]
            Ry = parameter[1]
            Rz = parameter[2]
            Tx = parameter[3]
            Ty = parameter[4]
            Tz = parameter[5]
            S = parameter[6]

            # Rotate part
            Rotate_Matrix_x =np.array([[1, 0, 0], [0, np.cos(Rx), -np.sin(Rx)], [0, np.sin(Rx), np.cos(Rx)]])
            Rotate_Matrix_y =np.array([[np.cos(Ry), 0, np.sin(Ry)], [0, 1, 0], [-np.sin(Rx), 0, np.cos(Rx)]])
            Rotate_Matrix_z =np.array([[np.cos(Rz), -np.sin(Rz), 0], [np.sin(Rz), np.cos(Rz), 0], [0, 0, 1]])
            Rotate_Matrix = Rotate_Matrix_x* Rotate_Matrix_y* Rotate_Matrix_z

            # Translate part
            T = [Tx, Ty, Tz]

            # Homogeneous
            Homogeneous_Rigid[0:3,0:3] = Rotate_Matrix * S
            Homogeneous_Rigid[0:3,3] = T

        # Define Rigid Transformation DOF = 7
        if self.length == 12:

            parameter = np.reshape(parameter,(3,4))
            Homogeneous_Rigid[0:3,0:4] = parameter
            
        self.Matrix = Homogeneous_Rigid   
        print(self.Matrix)    

    def rigid_transform (self, parameter = None):
        
        


        # Random affine generation
        if self.length == 0:
            return

        # Rigid transformation DOF = 6



    def affine_transform (self, parameter = None):
        

    

A = np.ones([3,4])
he = Affinetransform(A)

# print(np.cos([1, 0, 0; 0, np.cos(Rx), -np.sin(Rx); 0, np.sin(Rx), np.cos(Rx)]))

# print(np.ones([2,2]))