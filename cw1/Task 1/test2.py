# test range
import numpy as np


# kernal = (np.ones([3,3,3],dtype= 'int8'))
# kernal[0,1,1] = 1;kernal[1,0,1] = 1;kernal[1,2,1] = 1;kernal[1,1,0] = 1;kernal[1,1,2] = 1;kernal[2,1,1] = 1

# hhh = np.ones([5,5,5])
# # a = np.convolve(hhh ,kernal)
# for i in range(kernal):
#     print(i)
# kernal = np.zeros([3,3,3])
# kernal[0,1,1] = 1;kernal[1,0,1] = 1;kernal[1,2,1] = 1;kernal[1,1,0] = 1;kernal[1,1,2] = 1;kernal[2,1,1] = 1
# hh = np.ones([3,3,3])
# a = np.sum(kernal*hh)
b = np.ones([3,3,3])
b[2,2,2] = 5
b[2,2,1] = 5
a = b

# def jj (t):
#     t[1] = 3
#     return t
kai = np.array([1,2,3,4,5,6,7])
# h = np.where(kai == 5)
h = np.zeros([3])
# jj(a)

# print(h[2])

# print(np.size(h,axis = 0))
print(h)
