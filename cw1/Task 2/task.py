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

        # 
        x = Coordinate_Second - Coordinate_First
        y = Coordinate_Third - Coordinate_First

        # calculate normal vector
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

    print('shape' + str(Normal_vertices.shape))
    return Normal_vertices ,Normal_triangle_centres




# get data from marching_cube
info = marching_cubes(data)
vertex = info[0]
triangle = info[1]
Vertex_March_cube = info[2]

# get data from function
Vertex_Normal_Vector,triangle_normal_vector = surface_normals_np(vertex,triangle)

# Compare normal vectors from two implementatinos
# dot product from two implementation
sz = Vertex_Normal_Vector.shape
dot_product = [[] for row in range(sz[0])]
for i in range(sz[0]):
    dot_product[i] = np.dot(Vertex_Normal_Vector[i],Vertex_March_cube[i])












# plt.imshow(data[:,36,:])
# plt.show()

print('world')




