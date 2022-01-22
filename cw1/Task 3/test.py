import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
# def func(x, y):
#     return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

    

# grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

# rng = np.random.default_rng()
# points = rng.random((1000, 2))
# values = func(points[:,0], points[:,1])

# from scipy.interpolate import griddata
# grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
# grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
# grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')




# x = np.linspace(0,Range[0],Range[0]+1)
# y = np.linspace(0,Range[1],Range[1]+1)
# z = np.linspace(0,Range[2],Range[2]+1)
# xg, yg  = np.meshgrid(x, y, indexing='ij', sparse=True)
# self.Result2 = griddata(np.transpose(Result[0:2,:]), Result[3,:], (xg, yg ), method='linear')
# print(' finish Interpolation')
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
rng = np.random.default_rng()
x = rng.random(10) - 0.5
y = rng.random(10) - 0.5
z = np.hypot(x, y)
X = np.linspace(min(x), max(x))
Y = np.linspace(min(y), max(y))
X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
interp = LinearNDInterpolator(list(zip(x, y)), z)
Z = interp(X, Y)
plt.pcolormesh(X, Y, Z, shading='auto')
plt.plot(x, y, "ok", label="input point")
plt.legend()
plt.colorbar()
plt.axis("equal")
plt.show()

# [x,y,z,v] [4,n]
# for x in range(Range[0]):
#     for y in range(Range[1]):




