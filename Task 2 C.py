# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 22:35:01 2019

@author: user
"""
import numpy as np
import scipy.linalg as sp
import time
import matplotlib.pyplot as plt


def LU_lin_solver(A,b):
    '''Calculates the soultion to a linear equation using LU decomposition.'''
    lu,piv=sp.lu_factor(A)
    
    return sp.lu_solve((lu,piv),b)

def SVD_lin_solver(A,b):
    '''This function uses SVD decomposition to calculate the soultion to a linear equation.'''
    #Decomposes A into U,Sigma,VT where U and VT are orthogonal and sigma is the singularity matrix. 
    U,Sigma,VT=np.linalg.svd(A)
    UT=np.transpose(U)
    w=np.dot(UT,b)
    #np.linalg.svd returns sigma as a vector so need to diagonilise it
    S=np.diag(Sigma)
    c=np.linalg.solve(S,w)
    V=np.transpose(VT)
    x=np.dot(V,c)
    
    return x

def Lin_error(LS,A,b):
    '''Calculates the error of a linear soultion calculator using the fact that b-A^(-1)x=0.'''
    x=LS(A,b)
    
    return np.max(abs(b-np.dot(A,x)))


timeLU=[]
timeSVD=[]
N=[]

for n in range (2,1000):
    N+=[n]
    StartTime=0
    EndTime=0
    A=np.random.rand(n,n)
    b=np.random.rand(n)
    
    
    StartTime=time.time()
    LU_lin_solver(A,b)
    EndTime=time.time()
    timeLU+=[EndTime-StartTime]

    
    StartTime=time.time()
    SVD_lin_solver(A,b)
    EndTime=time.time()
    timeSVD+=[EndTime-StartTime]
    
plt.title('Run time against linear equation size')
plt.plot(N,timeLU,label='LU decomposition')
plt.plot(N,timeSVD,label='SDV decomposition')
plt.legend()
plt.ylabel('Time (s)')
plt.xlabel('N')
plt.show()
plt.clf()