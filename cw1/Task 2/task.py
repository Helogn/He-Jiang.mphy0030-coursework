import numpy as np
from numpy.lib.nanfunctions import nanprod
from skimage.measure import marching_cubes
import matplotlib.pyplot  as plt

data = np.load ('label_train00.npy')

def surface_normals_np(vertex, triangle):

    
    length_triangle = triangle.shape
    length_vertex = vertex.shape

    Normal_triangle_centres = np.zeros(length_triangle)
    

    # calculate triangle normal vector
    for i in range (length_triangle[0]):

        # get number of vertex
        First_point = triangle[i][0]
        Second_point = triangle[i][1]
        Third_point = triangle[i][2]

        # get coordinate of vertex
        Coordinate_First = vertex[First_point]
        Coordinate_Second = vertex[Second_point]
        Coordinate_Third = vertex[Third_point]
        # print('first ' + str(Coordinate_First) + ' second ' + str(Coordinate_Second) + ' Third ' +str(Coordinate_Third))

        # calculate normal vector
        x = Coordinate_Second - Coordinate_First
        y = Coordinate_Third - Coordinate_First
        # print('X: ' + str(x) + '   Y: '+ str(y))
        Normal_triangle_centres[i] = np.cross(x,y)


    # calculate vertex normal vector
    medium_normal = [[] for row in range(length_vertex[0])]
   

    # medium_normal is the vertex stored related triangle normal vector
    for j in range (length_triangle[0]):
        for i in range(3):
            medium_normal[triangle[j][i]].append(Normal_triangle_centres[j]) 


    # calculate vertex normal vector 
    medium_normal = np.array(medium_normal)
    
    Normal_vertices = [[] for row in range(length_vertex[0])]
    for i in range (length_vertex[0]):
        
        x = []
        y = []
        z = []
        # for j in range (shape_medium[1]):
        for j in range (len(medium_normal[i])):

            x.append(medium_normal[i][j][0])
            y.append(medium_normal[i][j][1])
            z.append(medium_normal[i][j][2])
        
        # Add Triangle Normal Vectors
        Nor = len(medium_normal[i])
        x = np.sum(np.array(x))/Nor
        y = np.sum(np.array(y))/Nor
        z = np.sum(np.array(z))/Nor
        
        # Normalization
        Normalization = (x*x + y*y + z*z) ** 0.5
        Normal_vertices[i] = [x/Normalization,y/Normalization,z/Normalization]

        Normal_vertices = np.array(Normal_vertices)

    print('successful')

    return Normal_vertices ,Normal_triangle_centres

info = marching_cubes(data)
vertex = info[0]
triangle = info[1]
march_nor = info[2]

vertex_normal_vector,triangle_normal_vector = surface_normals_np(vertex,triangle)


plt.imshow(data[:,36,:])
plt.show()

print('world')




