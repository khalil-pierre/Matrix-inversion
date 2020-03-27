# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:07:36 2019

@author: user
"""

import numpy as np
import scipy.linalg as sp
import time
import matplotlib.pyplot as plt
import random



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
                
def Gauss_Elimination(A):
    '''Calculates the inverse of a matrix using gauss jordan elimination.'''
    N=A.shape
    I=np.identity(N[0])
    
    for i in range(N[0]):
        for l in range(N[0]):
            #Uses pivoting to put largest element in the colloum in diagonal element for that row
            #This prevents breaking when diagonal element is zero.
            '''if l>i and abs(A[l,i])>abs(A[i,i]):
                
                RowA=[]
                RowB=[]
                
                for k in A[i]:
                    RowA.append(k)
                
                for j in I[i]:
                    RowB.append(j)
                    
                A[i]=A[l]
                A[l]=RowA
                
                I[i]=I[l]
                I[l]=RowB
                
            else:
                pass
            '''
        pivot=A[i,i]
        for z in range(N[1]):
            A[i,z]=A[i,z]/pivot
            I[i,z]=I[i,z]/pivot
            #Normalises each row
    
        for u in range(N[0]):
            #Subtracts each colloum apart from the diagonal by the diagonal multiplyed by a scale factor needed to
            #make all the non-diagonal elements zero
            if u!=i:
                scalefactor=A[u,i]
                A[u]-=scalefactor*A[i]
                I[u]-=scalefactor*I[i]
            
            else:
                pass
            
    return I


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
    c=np.linalg.solve(S,w)
    V=np.transpose(VT)
    x=np.dot(V,c)
    
    return x

def Lin_error(LS,A,b):
    '''Calculates the error of a linear soultion calculator using the fact that b-A^(-1)x=0.'''
    x=LS(A,b)
    
    return np.max(abs(b-np.dot(A,x)))

def Wire_vector(R_drum,r):
    '''Calculates the position vector that run along a wire. Uses the position of the drums and the position of the artist.'''
    return R_drum-r

def cosz(r):
    '''Calculates the cosine of the angle between the x-y plane and the wire vector.'''
    return np.sqrt((r[0]**2)+(r[1]**2))/np.linalg.norm(r)

def sinz(r):
    '''Calculates the sine of the angle between the x-y plane and the wire vector.'''
    return r[2]/np.linalg.norm(r)

def cosx(r):
    '''Calculates the component of the tension projected in the x-y plane in the x direction.'''
    return r[0]/np.sqrt((r[0]**2)+(r[1]**2))
    
def sinx(r):
    '''Calculates the component of the tension projected in the x-y plane in the y direction.'''
    return r[1]/np.sqrt((r[0]**2)+(r[1]**2))

def CoefficientMatrix(r):
    '''Calaculates the matrix of coefficients used to solve the linear equations givin in the problem sheet as a function of position.'''

    
    R_drum1=np.array([0,0,8])
    R_drum2=np.array([15,0,8])
    R_drum3=np.array([15/2,8,8])
    
    
    wire_vector1=Wire_vector(R_drum1,r)
    wire_vector2=Wire_vector(R_drum2,r)
    wire_vector3=Wire_vector(R_drum3,r)
    
    
    sinz1=sinz(wire_vector1)
    cosz1=cosz(wire_vector1)
    sinx1=sinx(wire_vector1)
    cosx1=cosx(wire_vector1)
    
    
    sinz2=sinz(wire_vector2)
    cosz2=cosz(wire_vector2)
    sinx2=sinx(wire_vector2)
    cosx2=cosx(wire_vector2)
    
    
    sinz3=sinz(wire_vector3)
    cosz3=cosz(wire_vector3)
    sinx3=sinx(wire_vector3)
    cosx3=cosx(wire_vector3)
    
    
    coefficient_matrix=np.zeros((3,3))
    
    coefficient_matrix[0]=np.array([cosz1*cosx1,cosz2*cosx2,cosz3*cosx3])
    coefficient_matrix[1]=np.array([cosz1*sinx1,cosz2*sinx2,cosz3*sinx3])
    coefficient_matrix[2]=np.array([sinz1,sinz2,sinz3])
    
    return coefficient_matrix


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