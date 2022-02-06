import numpy as np
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter


def surface_normals_np(vertex, triangle):

#   takes a triangulated surface, represented by a list of vertices and a list of triangles, as input
#   Input: a list of vertices 
#          a list of triangles
#   Output: normal vector at vertices
#           normal vector at triangle centres   
    
    
    shape_triangle = triangle.shape
    shape_vertex = vertex.shape

    Normal_triangle_centres = np.zeros(shape_triangle)

    # calculate triangle normal vector
    for i in range (shape_triangle[0]):

        # get index of vertex from triangle 
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

        # calculate triangle normal vector
        Normal_triangle_centres[i] = np.cross(x,y)


    # calculate vertex normal vector
    medium_normal = [[] for row in range(shape_vertex[0])]
   

    # medium_normal is the vertex stored related triangle normal vector
    for j in range (shape_triangle[0]):
        for i in range(3):
            medium_normal[triangle[j][i]].append(Normal_triangle_centres[j]) 


    # calculate vertex normal vector 
    medium_normal = np.array(medium_normal)
    
    Normal_vertices = [[] for row in range(shape_vertex[0])]
    for i in range (shape_vertex[0]):
        
        x = []
        y = []
        z = []

        # for j in range (shape_medium[1]):
        for j in range (len(medium_normal[i])):

            x.append(medium_normal[i][j][2])
            y.append(medium_normal[i][j][1])
            z.append(medium_normal[i][j][0])
        # Add Triangle Normal Vectors
        Nor = len(medium_normal[i])
        x = np.sum(np.array(x))/Nor
        y = np.sum(np.array(y))/Nor
        z = np.sum(np.array(z))/Nor
        
        # Normalization
        Normalization = (x*x + y*y + z*z) ** 0.5
        Normal_vertices[i] = [z/Normalization,y/Normalization,x/Normalization]

    Normal_vertices = np.array(Normal_vertices)

    # print('shape' + str(Normal_vertices.shape))
    return Normal_vertices ,Normal_triangle_centres

data = np.load ('label_train00.npy')
# get data from marching_cube
vertex,triangle,Vertex_March_cube,info = marching_cubes(data,spacing=(2,0.5,0.5))

# calculate normal vectors
Vertex_Normal_Vector,triangle_normal_vector = surface_normals_np(vertex,triangle)

# Compare normal vectors from two implementatinos
# dot product from two implementation
sz = Vertex_Normal_Vector.shape
dot_product = [[] for row in range(sz[0])]
for i in range(sz[0]):
    dot_product[i] = np.dot(Vertex_Normal_Vector[i],Vertex_March_cube[i])




# compare the vertex normal vectors computed from the two implementations
print('\n\nshape of Vertex result from marching_cubes      ' + str(Vertex_March_cube.shape))
print('shape of Vertex result from surface_normals_np  ' + str(Vertex_Normal_Vector.shape))
print('\nstandard of subtraction between Vertex_Normal_Vector and Vertex_March_cube   \n'  + str(format(np.std(dot_product),'.5f')))
print('mean of Vertex_March_cube     ' + str(format(np.mean(Vertex_March_cube),'.5f')))
print('mean of Vertex_Normal_Vector  ' + str(format(np.mean(Vertex_Normal_Vector),'.5f')))
print('percentage of different part  ' + str(format((np.mean(Vertex_Normal_Vector) - np.mean(Vertex_March_cube))/np.mean(Vertex_March_cube),'.3f')) + '%')

# Comment: Vertex_March_cube and Vertex_Normal_Vector have same length of vector
#          the mean and standard of both variants are closest. 
#          my method is to calculate mean of normal vector at triangle centre to get vertex normal vector
#          However the precise resolution should consider angles among vectors at triangle centre.
#          This might be main difference between these two methods.


# Design a method to compare the vertex normal and the triangle-centre normal vectors,
# computed from surface_normals_np, and use it to compare their difference and comment on the results

mean_Vertex = format(np.mean(Vertex_Normal_Vector),'.8f')
mean_triangle = format(np.mean(triangle_normal_vector),'.8f')
std_Vertex = format(np.std(Vertex_Normal_Vector),'.8f')
std_triangle = format(np.std(triangle_normal_vector),'.8f')
print('\n mean of vertex and triangle vertex: ' + str(mean_Vertex) + 'triangle: ' + str(mean_triangle))
print('\n standard of vertex and triangle vertex: ' + str(std_Vertex) + 'triangle: ' + str(std_triangle))

# Comment: mean of triangle normal vector is 0, however mean of vertex normal vector is not 0
#          standard of triangle is smaller than vertex
#          And vertex normal vector is calculated by meaning triangle normal vector
#          triangle normal vector is symmetric in space level


# Use 3D Gaussian filter to smooth the binary segmentation
print('\nOriginal number of Label voxel which value is -1- is ' + str(np.sum(data)))
print('----------------------------------------------------------------------------\n\n')
Sigma = [0.1,0.15,0.2,0.25,0.4,0.5,0.6,0.7,0.8,1,2]
for i in (Sigma):
    Filtered_image = gaussian_filter(data, sigma = i)
    print('\nsigma: ' + str(i) + '  Sum of Label Voxel: ' + str(np.sum(Filtered_image)))
    if np.sum(Filtered_image) != 0:
        vertex,triangle,Vertex_March_cube,info  = marching_cubes(Filtered_image)
        print(Vertex_March_cube.shape)

        # calculate normal vectors
        Vertex_Normal_Vector,triangle_normal_vector = surface_normals_np(vertex,triangle)
        print('shape of Vertex result from marching_cubes      ' + str(Vertex_March_cube.shape))
        print('shape of Vertex result from surface_normals_np  ' + str(Vertex_Normal_Vector.shape))
        print('\nstandard of subtraction between Vertex_Normal_Vector and Vertex_March_cube  is '  + str(format(np.std(dot_product),'.5f')))
        print('mean of Vertex_March_cube     ' + str(format(np.mean(Vertex_March_cube),'.5f')))
        print('mean of Vertex_Normal_Vector  ' + str(format(np.mean(Vertex_Normal_Vector),'.5f')))
        print('percentage of different part  ' + str(format((np.mean(Vertex_Normal_Vector) - np.mean(Vertex_March_cube))/np.mean(Vertex_March_cube),'.3f')) + '%')

# about impart of sigma:
# Comment :
#           as sigma increases from 0 to 1, an increasing number of voxels convert from 1 to 0.
#           Obviously, amount of voxel with value 1 decreases. What we expected to see is smoothing image with mean and std decreasing.
#           However, mean and standard do not change apparently. 
#           I think it is due to property of label. This image have only two values ( 0 and 1), it does not reflect smoothing properly
#           And in discrete space, smoothing is not as good as in continuous space.
#           both above properties affect final results.







