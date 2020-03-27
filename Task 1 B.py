# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 21:39:51 2019

@author: user
"""
import numpy as np


Matrix=np.array([[1,1,1],[1,2-1],[2,3,1]])
b=np.array([5,10,15])
M=Matrix.shape


I=np.identity(M[0])



for i in range(M[0]):
    
    for l in range(M[0]):
        if l>i and abs(Matrix[l,i])>abs(Matrix[i,i]):
            A=[]
            B=[]
            for j in Matrix[i]:
                A.append(j)
            
            for k in I[i]:
                B.append(k)

            Matrix[i]=Matrix[l]
            Matrix[l]=A
            
            I[i]=I[l]
            I[l]=B
        
        else:
            pass
            
        print(b)
    
  
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


print(I)

