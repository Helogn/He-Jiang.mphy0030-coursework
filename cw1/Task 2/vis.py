import numpy as np
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
# import matplotlib.pyplot  as plt
from task import surface_normals_np
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

data = np.load ('label_train00.npy')
# get data from marching_cube
vertex, triangle, Vertex_March_cube, info = marching_cubes(data,spacing =(2,0.5,0.5))

# calculate normal vectors
Vertex_Normal_Vector,triangle_normal_vector = surface_normals_np(vertex,triangle)
Vertex_Normal_Vector = np.transpose(Vertex_Normal_Vector)
sz = Vertex_Normal_Vector.shape

# calculate origin of centre vector
sz2 = triangle.shape
ori_centre_vec = np.zeros(sz2)
for i in range(sz2[0]):
    for j in range(3):
        First_point = triangle[i][0]
        Second_point = triangle[i][1]
        Third_point = triangle[i][2]
        ori_centre_vec[i] = np.array((vertex[First_point] + vertex[Second_point] + vertex[Third_point])/3)

# -----------------plot/save part---------------------

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')

# mesh = Poly3DCollection(vertex[triangle])
# mesh.set_edgecolor('black')
# mesh.set_facecolor('white')
# ax.add_collection3d(mesh)

# # plot vertex normal vector
# ax.quiver(np.transpose(vertex[:,0]), np.transpose(vertex[:,1]), np.transpose(vertex[:,2]), Vertex_Normal_Vector[0,:], Vertex_Normal_Vector[1,:], Vertex_Normal_Vector[2,:], length=2, normalize=True, color = 'blue',alpha = 0.05)
# # plot triangle normal vector
# ax.quiver(np.transpose(ori_centre_vec[:,0]), np.transpose(ori_centre_vec[:,1]), np.transpose(ori_centre_vec[:,2]), np.transpose(triangle_normal_vector[:,0]), np.transpose(triangle_normal_vector[:,1]), np.transpose(triangle_normal_vector[:,2]), length = 2, normalize=True, color = 'red',alpha = 0.05)

# # set environment
# ax.set_xlabel("x-axis")
# ax.set_ylabel("y-axis")
# ax.set_zlabel("z-axis")

# ax.set_xlim(10, 60)  # a = 6 (times two for 2nd ellipsoid)
# ax.set_ylim(0,50)  # b = 10
# ax.set_zlim(0, 55)  # c = 16
# ax.set_title('Red is triangle vector, Blue is vertex vector ')

# plt.tight_layout()
# plt.show()
