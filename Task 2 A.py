# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 21:13:33 2019

@author: user
"""

import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt
import time


def Minor(A,i,j):
    '''calculates the matrix of minors for a NxN matrix, A, around some matrix element i,j.'''
    #For calculating the matrix of cofactors and for calculating the determinate of a N>2 matrix
    #I will need to be able to calculate a matrix of minors
    N=A.shape[0]
    minor=np.zeros((N-1,N-1))
    cork=0
    #As we move past k=i I need to correct the k values using cork.
    #This is so that elements in the NxN matrix will corrospond to a position in the minor matrix.
    for k in range(N):
        if k==i:
            cork=1
        else:
            corh=0
            for h in range(N):
                if h==j:
                    corh=1
                else:
                    minor[k-cork][h-corh]=A[k][h]
  
    return minor


def Det(A):
    '''Calculates the determinant of an NxN matrix, A.'''
    N=A.shape[0]
    #Only going to use square matrices so should just be able to take first element
    if N==2:
        return A[0,0]*A[1,1]-A[0,1]*A[1,0]
        
    elif N>2:
        #For n>2 we need to create a matrix of minors.
        #The code will have to keep looping until the NxN matrix is reduced to a series of 2x2 matrices.
        det=0
        for i in range(N):
            det+=((-1)**i)*A[0][i]*Det(Minor(A,0,i))
            
        
        return det 

def Cofactors(A,Transpose=True):
    '''Returns the Transpose of the cofactors for a NxN matrix, A, if \n
    Transpose=True. If Trasnpose=False then the function will return the matrix of cofactors'''
    N=A.shape[0]
    cofactors=np.zeros((N,N))
    #When calculating the determinate we want the inverse of the cofactors.
    #So unless specified otherwise this function will return the inverse of the cofactors 
    for i in range(N):
        for j in range(N):
            cofactors[i][j]= ((-1)**(i+j))*Det(Minor(A,i,j))
    
    if Transpose==True:
        transpose=np.zeros((N,N))
        for k in range(N):
            for l in range(N):
                transpose[l,k]=cofactors[k,l]
        return transpose 
    
    else:
        return cofactors 

def Invert(A):
    '''Calculates the inverse of a matrix A.'''
    N=A.shape[0]
    if N==2:
        invert=(1/Det(A))*np.array([[A[1,1],-A[0,1]],[-A[1,0],A[0,0]]])
    
    else:
        invert=(1/Det(A))*Cofactors(A,True)
    
    return invert

def Lin_solver(A,b):
    '''Calculates the soultion to a linear equation using Cramer's rule.'''
    return np.dot(Invert(A),b)

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

b=np.array([5,10,15])


timeLU=[]
timeSVD=[]
K=[]
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



Error=[]
ErrorLU=[]
ErrorSVD=[]
K=[]

for i in range(1,10000):
    k=(1e-18)*i
    K+=[k]
    A=np.array([[1,1,1],[1,2,-1],[2,3,k]])
    b=np.array([5,10,15])
    
    #Error+=[Lin_error(Lin_solver,A,b)]
    ErrorLU+=[Lin_error(LU_lin_solver,A,b)]
    ErrorSVD+=[Lin_error(SVD_lin_solver,A,b)]
    

plt.title('Numerical error as linear equation approaches singularity')
#plt.plot(K,Error,label='Cramers rule')
plt.plot(K,ErrorLU,label='LU decomposition')
plt.plot(K,ErrorSVD,label='SVD decomposition')
plt.ylabel('Numerical error')
plt.xlabel('K value')
#plt.yscale('log')
#plt.xscale('log')
plt.legend()
plt.show()
plt.clf()

    
    

'''


for i in range(10):
    k=(10-i)*10**(-16)
    K+=[k]
    Matrix=np.array([[1,1,1],[1,2,-1],[2,3,k]])
    
    ErrorC+=[]
    ErrorLU+=[]
    ErrorSVD+=[]
    
plt.title('Error of LU decomposition as matrix approaches singularity')
plt.plot(K,ErrorLU)
plt.ylabel('Numerical error')
plt.xlabel('Value of k')
plt.show()
plt.clf()

'''

