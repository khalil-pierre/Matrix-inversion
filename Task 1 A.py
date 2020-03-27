# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 16:45:05 2019

@author: user
"""
import numpy as np


#Matrix to use to show program can calcuate inverse of 4x4 matrices
Matrix=np.array([[1.0,0.0,0.0,1.0],[0.0,2.0,1.0,2.0],[2.0,1.0,0.0,1.0],[2.0,0.0,1.0,4.0]])
#Matrix=np.array([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]])
M=Matrix.shape



#print(Matrix.max(axis=1))

I=np.identity(M[0])


x=0
for i in range(M[0]):
    pivot=Matrix[i,i]
    for j in range(M[1]):
        Matrix[i,j]=Matrix[i,j]/pivot
        I[i,j]=I[i,j]/pivot
        #Normalises each row
    
    for k in range(M[0]):
        if k!=i:
            scalefactor=Matrix[k,i]
            Matrix[k]-=scalefactor*Matrix[i]
            I[k]-=scalefactor*I[i]
            
        else:
            pass

A=np.array([[1,2,3],[4,5,6],[7,8,-9]])
B=np.array([[1,2,3],[4,5,6]])

print(np.max(abs(A)))
print(np.random.rand(5,5))

'''
print(I)

print(np.allclose(np.identity(M[0]),Matrix))
print(len(Matrix))
'''
#Can append an item using Matrix.append()
   
     