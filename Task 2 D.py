# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 15:22:40 2019

@author: user
"""

import numpy as np 
import matplotlib.pyplot as plt
import scipy.linalg as sp


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
    '''Returns the Transpose of the cofactors for a NxN matrix, A, if 
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

def Matrix_multiplication(A,B):
    '''Multiples two matrices A and B together. Matrices do not commute so make \n
    sure the order of the matrices are correct.'''
    AShape=A.shape
    BShape=B.shape
    Product=np.zeros((AShape[0],BShape[1]))
    #When muliplying two matrixs mxn and nxp the new matrix that is formed is given by mxp 
    for i in range(AShape[0]):
        for j in range(BShape[0]):
            for k in range(len(B)):
                Product[i][j]+=A[i][k]*B[k][j]
        
    return Product
                
def Max_error(inv,A):
    '''Finds the error of a function that calculates the inverse of a matrix.'''
    '''The argument inv is the function I want to test and A is the matrix I am''' 
    '''using to test the function. This function calculates the inverse of a matrix'''
    '''then it mulitples the matrix by the inverse the product should be equal to the''' 
    '''identity matrix. This function calculates the difference between the product and''' 
    '''the identity and takes the largest element as the error'''
    
    N=A.shape[0]
    Identity=np.identity(N)
    B=inv(A)
    ide=Matrix_multiplication(A,B)
    diff=ide-Identity
    return np.max(abs(diff))

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
    c=sp.solve(S,w)
    V=np.transpose(VT)
    x=np.dot(V,c)
    
    return x

def Lin_error(LS,A,b):
    '''Calculates the error of a linear soultion calculator using the fact that b-A^(-1)x=0.'''
    x=LS(A,b)
    
    return np.max(abs(b-np.dot(A,x)))







































