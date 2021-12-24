# def mymax(a:str = "12345",b:str = "123456789"):
#     return max(len(a),len(b))
# def mymax(a :int = 0,b:int = 1):
#     return max(a,b)

# print(mymax("one","three"))
# print(mymax(1,3))#AllArgs
# print(mymax())#NoArgs
import numpy as np
class hhh:

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


        if length == 7:
            return
        if length == 12:
            return

    
he = hhh()
A = np.ones([2,6])
he.rigid_transform(A)
# print(np.cos([1, 0, 0; 0, np.cos(Rx), -np.sin(Rx); 0, np.sin(Rx), np.cos(Rx)]))

print(np.ones([2,2]))