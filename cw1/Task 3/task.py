from cmath import nan
from inspect import Traceback
import numpy as np
import matplotlib.pyplot  as plt
from scipy.interpolate import griddata,LinearNDInterpolator
from PIL import ImageFilter, Image

class Image3D :

    def __init__(self,Array,dims =[1,1,1] ):


        self.Array = Array
        self.dims = dims
        
        sz = Array.shape
        self.sz = sz
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
        # print('max = ' + str(np.max(Coordinate_Matrix,1)))
        # print('min = ' + str(np.min(Coordinate_Matrix,1)))
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
        # self.Result2 = np.floor(np.zeros([Range[0]+1,Range[1]+1,Range[2]+1],dtype='int32')-1000)
        # print(sz_Result[1])
        # for N in range (sz_Result[1]):

        #     tx = np.int16(np.floor(Result[0,N])-self.matrix[0,3])
        #     ty = np.int16(np.floor(Result[1,N])-self.matrix[1,3])
        #     tz = np.int16(np.floor(Result[2,N])-self.matrix[2,3])
        #     self.Result2[tx,ty,tz] = (Result[3,N])

        # print('finish Transfor')

        #long running
        #do something other

        # --------------------------------------------------------------------------
        # # interpolation        
        # x = np.linspace(0,Range[0],Range[0]+1)
        # y = np.linspace(0,Range[1],Range[1]+1)
        # z = np.linspace(0,Range[2],Range[2]+1)
        # # xg, yg, zg  = np.meshgrid(x, y,z, indexing='ij', sparse=True)
        # xg, yg, zg  = np.meshgrid(x, y,z)
        # # self.Result2 = griddata(np.transpose(Result[0,:]), Result[3,:], (xg,yg ), method='linear')
        # # X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
        # interp = LinearNDInterpolator(list(zip(Result[0,:],Result[1,:],Result[2,:])), Result[3,:])
        
        # print(' finish Interpolation')

        # self.Result2 = interp(xg, yg, zg)
        # print(self.Result2.shape)
        # ----------------------------------------------------------------------------

        # ------------------------------------------ test 2-------------------------
        
        # x = Result[0,:].reshape(self.sz)
        # y = Result[1,:].reshape(self.sz)
        # z = Result[2,:].reshape(self.sz)
        # v = Result[3,:].reshape(self.sz)
        # x = np.linspace(0,Range[0],Range[0]+1)
        # y = np.linspace(0,Range[1],Range[1]+1)
        # z = np.linspace(0,Range[2],Range[2]+1)
        # gx,gy,gz = np.mgrid[0:Range[0]:Range[0]+1j,0:Range[1]:Range[1]+1j,0:Range[2]:Range[2]+1j]
        # # interp = griddata(np.array([np.reshape(x,-1),np.reshape(y,-1),np.reshape(z,-1)]).T, np.reshape(v,-1))
        # interp = griddata(np.transpose (Result[0:3,:]), np.transpose(Result[3,:]) ,(gx,gy,gz) ,method='linear')
        # print(x.shape)

        # ---------------------------- test 3 -------------------------------------
        # x = Result[0,:].reshape(self.sz)
        # y = Result[1,:].reshape(self.sz)
        # z = Result[2,:].reshape(self.sz)
        # v = Result[3,:].reshape(self.sz)

        # Output_x = x[7,:,:].reshape(-1); x_max = np.max(Output_x); x_min = np.min(Output_x)
        # Output_y = y[7,:,:].reshape(-1); y_max = np.max(Output_y); y_min = np.min(Output_y)
        # # Output_z = z[7,:,:].reshape(-1); z_max = np.max(Output_z); z_min = np.min(Output_z)      
        # Output_v = v[7,:,:].reshape(-1)     
        # x = np.linspace(x_min,(x_max),np.int16(x_max-x_min)+1)
        # y = np.linspace((y_min),(y_max),np.int16(y_max-y_min)+1)
        # # z = np.linspace((z_min),(z_max),np.int16(z_max-z_min)+1)
        # # [X, Y ,Z] = np.meshgrid(x,y,z)
        # X, Y  = np.meshgrid(x,y)
        # # pix_coords = np.array([np.reshape(X, -1) ,np.reshape(Y ,-1),np.reshape(Z ,-1)]).T
        # pix_coords = np.array([np.reshape(X, -1) ,np.reshape(Y ,-1)]).T
        # A = np.array([Output_x,Output_y]).T
        # B = np.squeeze(np.array([X,Y])).T
        # interp = griddata(A, (Output_v) ,(X,Y) ,method='linear')
        # --------------------- test4 ----------------------

        x = Result[0,:].reshape(self.sz)
        y = Result[1,:].reshape(self.sz)
        z = Result[2,:].reshape(self.sz)
        v = Result[3,:].reshape(self.sz)
        x_max = np.max(Result[0,:]); x_min = np.min(Result[0,:])
        y_max = np.max(Result[1,:]); y_min = np.min(Result[1,:])
        z_max = np.max(Result[2,:]); z_min = np.min(Result[2,:])
        
        x1 = np.linspace(x_min,(x_max),np.int16(x_max-x_min)+1)
        y1 = np.linspace((y_min),(y_max),np.int16(y_max-y_min)+1)
        X, Y  = np.meshgrid(x1,y1)
        print(len(x1))
        sz = x.shape
        output = np.zeros(np.int16([sz[0],len(y1),len(x1)]))
        for A in range(sz[0]):
            Output_x = x[A,:,:].reshape(-1)
            Output_y = y[A,:,:].reshape(-1)
            Output_v = v[A,:,:].reshape(-1)     
            # z = np.linspace((z_min),(z_max),np.int16(z_max-z_min)+1)
            # [X, Y ,Z] = np.meshgrid(x,y,z)
            
            # pix_coords = np.array([np.reshape(X, -1) ,np.reshape(Y ,-1),np.reshape(Z ,-1)]).T
            pix_coords = np.array([np.reshape(X, -1) ,np.reshape(Y ,-1)]).T
            point = np.array([Output_x,Output_y]).T
            B = np.squeeze(np.array([X,Y])).T
            interp = griddata(point, (Output_v) ,(X,Y) ,method='linear')
            # print(interp.shape)
            output[A,:,:] = np.array(interp) # Z Y X
            # print('finish one slice')
        output[np.isnan(output)] = 0
        for A in range(1,sz[0]-1):
            output[A,:,:] = output[A-1,:,:]/6 + output[A+1,:,:]/6 + output[A,:,:] * 4 /6

        # 



        return output


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

        print('parameter of matrix = \n' + str(self.trans))


    def rigid_transform (self, parameter ):
        
        # judge data type

        length = len(parameter)

        # judge length of vector
        if (length in [6,7,12]) == False:
            print('Error Input Vertor Length')
            return

        # Random affine generation

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

    def affine_transform (self, par):
        
        # judge data type
        length = len(par)

        Homogeneous = np.zeros([4,4])

        if length == 12:

            Homogeneous[0:3,0:3] = [[par[0],par[1],par[2]],[par[3],par[4],par[5]],[par[6],par[7],par[8]]]
            Homogeneous[0:3,3] = [par[9],par[10],par[11]]
            Homogeneous[3,3] = 1

            return Homogeneous

    def random_transform_generator (self,strength = 1):
        # strength from 1 to higher value ->  small change to super change

        Matrix = np.zeros([4,4])
        n, m = 3, 4

        H = np.random.rand(n, m)
        u, s, vh = np.linalg.svd(H, full_matrices=False)
        u = np.round(u,4)
        
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



        return Matrix


Image = np.load('image_train00.npy')
obj = Image3D(Image)
# Transformation = Affinetransform([0,10,0,2,0,0,2])
Transformation = Affinetransform()
print(Transformation.random_transform_generator(3))
# print(Transformation.affine_transform())
# Transformation = Affinetransform()
result = obj.warp(Transformation)

# print(result.shape)
plt.subplot(2, 2, 1)
plt.imshow(Image[10,:,:])
plt.hot()
plt.subplot(2, 2, 2)
plt.imshow(result[10,:,:])
# plt.savefig('test.png')

plt.subplot(2, 2, 3)
plt.imshow(result[15,:,:])

plt.subplot(2, 2, 4)
plt.imshow(result[20,:,:])

plt.show()

