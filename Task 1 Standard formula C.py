# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 15:45:00 2019

@author: user
"""
import numpy as np

def Det2(A):
    if A.shape==(2,2):
        det2=A[0,0]*A[1,1]-A[0,1]*A[1,0]
        
        return det2
    else:
        return print('Input is not a 2x2 matrix')
    
Matrix=np.array([[1,2],[1,1]])

print(Det2(Matrix))
'''
def Inverse(A):
    '''
M=np.array([[1,6,3],[4,5,6],[-11,8,9]])
print(M.argmax(axis=1))    
    
    
    