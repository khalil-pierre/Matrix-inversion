# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:25:50 2019

@author: user
"""
#The first task is to create a program that inverts a matrix using the standard fomula.
#There are 4 things we need to be able to calcualte:
#The minor matrix of a element 
#The determinate of a NxN matrix
#The cofactors of a NxN matrix
#The transpose of a Nxn matrix

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


MyInput='0'
while MyInput != 'q':
    MyInput=input('Enter a task "1","2","3" or "q" to quit: ')
    if MyInput=='1':
        Matrix=0
        print('You have choosen task 1.'
              ' In task 1 I have been asked to calculate the inverse of a matrix.')
        
        
        Size=int(input('Please select the size of the square matrix you are intrested in: '))
        
        T =input('Please select the matrix you are intrested in'
                   ' "I" for an identity matrix, "R" for a random matrix'
                   ' or "C" for your own matrix: ')

        if T == 'I' or T=='i':
            Matrix=np.identity(Size)
        elif T == 'R' or T=='r':
            Matrix=np.random.rand(Size,Size)
        
        elif T == 'C' or T=='c':
            Matrix=np.zeros((Size,Size))
            for i in range(Matrix.shape[0]):
                for j in range(Matrix.shape[0]):
                    Matrix[i][j]=float(input('Please enter a value for the element:'))
                    print(Matrix)
        
        else:
            print(T, ' is not a valid choice')
        
        print()
        print('The inverse of the matrix')
        print()
        print(Matrix)
        print()
        print('is...')
        print()
        print(Invert(Matrix))
        print('The error is=', Max_error(Invert,Matrix))
        print()
        print('In task 1 I was also asked to find the relationship' 
              ' between the matrix size and the inversion time of my program.'
              ' Please wait whilst the graph for the inversion time against matrix' 
              ' size is plotted.')
        
        size=[]
        T=[]
        
        
        for N in range(2,10):
            size+=[N]
            matrix=np.random.rand(N,N)
            StartTime=time.time()
            Invert(matrix)
            EndTime=time.time()
            T+=[EndTime-StartTime]
            
        
        plt.title('Inversion time against matrix size')
        plt.plot(size,T)
        plt.ylabel('Inversion time (ln(t))')
        plt.xlabel('Matrix size')
        plt.yscale('log')
        plt.show()
        plt.clf()
        
        
        print('I was also interested how the numerical error varies with matrix size.'
              ' Please wait whilst the graph is generated.')
        
        
        #Could calculate both graphs in same for loop but it takes time
        size=[]
        Error=[]
        
        for n in range(2,7):
            error=0
            count=0
            size+=[n]
            
            for av in range(500):
                
                
                #At first I thought I would just be able to use the np.random.rand function.
                #However the size of the error is dominated by floating point error so for 
                #smaller value elements the error will be larger and I will not be able to 
                #see how the error depends on the size of the matrix. By using the 
                #randit function I have some control over the magnitude of the numbers envolved.
                #However there is a greater chance that singular matrices will be produced.
            
                matriX=np.zeros((n,n))
            
                for i in range(n):
                    for j in range(n):
                        matriX[i][j]=random.randint(-10,10)
                
                if round(Det(matriX),5)==0.0:
                    pass
                
                else:
                    count+=1
                    error+=Max_error(Invert,matriX)
            
            Error+=[error/count]
            
        
        
        plt.title('Numerical error against matrix size')
        plt.plot(size,Error)
        plt.ylabel('Numerical error')
        plt.xlabel('Matrix size')
        plt.show()
        plt.clf()
        
        
        
        
            
    elif MyInput=='2': 
        print('You have choosen task 2')
        
        print('In task 2 we were asked to investgate how LUD and SUD decomposition methods scaled'
              ' with the size of the linear equation.')
        print('Please wait whilst the plot of linear equation size against soultion time is generated.')
        
        b=np.array([5,10,15])

        #ErrorC=[]
        #ErrorLU=[]
        #ErrorSVD=[]
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
        plt.plot(N,timeSVD,label='SVD decomposition')
        plt.legend()
        plt.ylabel('Time (s)')
        plt.xlabel('N')
        plt.show()
        plt.clf()

        print('I was also interested in how the soultions behave as the matrix of coefficients'
              ' approaches singularity.')
        
        Error=[]
        ErrorLU=[]
        ErrorSVD=[]
        K=[]

        for i in range(10000):
            k=(1e-18)+(1e-18)*i
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
        plt.xscale('log')
        plt.legend()
        plt.show()
        plt.clf()
    

  
    
        
    elif MyInput=='3':
        print('You have choosen task 3')
        print('In task 3 we were asked to solve a linear physics problem using an appropriate algorithm.'
              ' Fist I looked at the 2D case where a trapeze artist was suspended above a stage with two wires.'
              ' I have calculated the tension in the wires as a function of position.')
       
        
        b=np.array([0,70*9.81])

        T1=[]
        T2=[]
        List1=[]
        List2=[]
        
        T1max=0
        T2max=0
        
        X1max=0
        Y1max=0
        
        X2max=0
        Y2max=0
        
        for y in np.linspace(0,7,100):
            for x in np.linspace(0,15,100):
                
                cos1=x/np.sqrt((x**2)+((8-y)**2))
                sin1=(8-y)/np.sqrt((x**2)+((8-y)**2))
                
                cos2=(15-x)/np.sqrt(((15-x)**2)+((8-y)**2))
                sin2=(8-y)/np.sqrt(((15-x)**2)+((8-y)**2))
                
                Matrix_coefficents=np.array([[cos1,-cos2],[sin1,sin2]])
                
                Tension2d=SVD_lin_solver(Matrix_coefficents,b)
                
                List1.append(Tension2d[0])
                List2.append(Tension2d[1])
                
                if Tension2d[0]>T1max:
                    
                    T1max=Tension2d[0]
                    X1max=x
                    Y1max=y
                    
                    
                elif Tension2d[1]>T2max:
                    
                    T2max=Tension2d[1]
                    X2max=x
                    Y2max=y
                    
                else:
                    pass
            
            T1.append(List1)
            T2.append(List2)
            List1=[]
            List2=[]
                
                        
        plt.title('Tension in wire 1')        
        plt.imshow(T1,origin='lower',extent=[0,15,0,7])
        plt.colorbar(label='Tension (N)')
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.show()
        plt.clf()
        
        plt.title('Tension in wire 2')
        plt.imshow(T2,origin='lower',extent=[0,15,0,7])
        plt.colorbar(label='Tension(N)')
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.show()
        plt.clf()
        
        print('The maximum tension in wire 1 is = ', str(T1max), 'N')
        print('The position of the maximum tension is x = ', str(X1max), 'm'
              ' and y = ', str(Y1max), 'm.')
        print()
        print('The maximum tension in wire 2 is = ', str(T2max), 'N')
        print('The position of the maximum tension is x = ', str(X2max), 'm'
              ' and y = ', str(Y2max), 'm.')
        print()
        print()
        print()
        
        print('If a 3rd wire is attached to the artist they can now move'
              ' backwards and forward as well as up and down.'
              ' If you select a height for the artist a crosssection of the tension'
              ' as a function of position for said height will be generated.')
        
        
        z=float(input('Select the height: '))

        Tension_wire_1=[]
        Tension_wire_2=[]
        Tension_wire_3=[]
        
        
        List1_3D=[]
        List2_3D=[]
        List3_3D=[]
        b=np.array([0,0,70*9.81])
        
        for y in np.linspace(1e-6,8,100):
            for x in np.linspace(1e-6,15,100):
                if y<(8/7.5)*x and y<(-8/7.5)*x+16:
                    
                    position=np.array([x,y,z])
                    
                    
                    Tension3D=LU_lin_solver(CoefficientMatrix(position),b)
                    
                    List1_3D.append(Tension3D[0])
                    List2_3D.append(Tension3D[1])
                    List3_3D.append(Tension3D[2])
        
                else:
                    
                    List1_3D.append(0.0)
                    List2_3D.append(0.0)
                    List3_3D.append(0.0)
                    
            
            Tension_wire_1.append(List1_3D)
            Tension_wire_2.append(List2_3D)
            Tension_wire_3.append(List3_3D)
                         
        
            List1_3D=[]
            List2_3D=[]
            List3_3D=[]
            
            
        plt.title('Tension in wire 1 in the x-y plane')      
        plt.imshow(Tension_wire_1,origin='lower',extent=[0,15,0,7])
        plt.colorbar(label='Tension (N)')
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.show()
        plt.clf()
        
        plt.title('Tension in wire 2 in the x-y plane')      
        plt.imshow(Tension_wire_2,origin='lower',extent=[0,15,0,7])
        plt.colorbar(label='Tension (N)')
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.show()
        plt.clf()
        
        plt.title('Tension in wire 3 in the x-y plane')      
        plt.imshow(Tension_wire_3,origin='lower',extent=[0,15,0,7])
        plt.colorbar(label='Tension (N)')
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.show()
        plt.clf()
        
        
        maxT1=0
        maxT2=0
        maxT3=0
        
        
        Max_position1=0
        Max_position2=0
        Max_position3=0
        
        
        for z in np.linspace(1e-6,7,50):
            for y in np.linspace(1e-6,8,50):
                for x in np.linspace(1e-6,15,50):
                    if y<(8/7.5)*x and y<(-8/7.5)*x+16:
                        
                        position=np.array([x,y,z])
                        
                        
                        Tension3D=LU_lin_solver(CoefficientMatrix(position),b)
                        
                        if Tension3D[0]>maxT1:
                            maxT1=Tension3D[0]
                            Max_position1=position
        
        
                        else:
                            pass
                        
                        if Tension3D[1]>maxT2:
                            maxT2=Tension3D[1]
                            Max_position2=position
           
                            
                        else:
                            pass
                        
                        if Tension3D[2]>maxT3:
                            maxT3=Tension3D[2]
                            Max_position3=position
        
                        else:
                            pass
                    
                    else:
                        pass
                        
        
        print('Max tension in wire 1 is = ', str(maxT1), ('N'))
        print('The max tension occurs at the position = ', str(Max_position1), 'm')
        print()
        print('Max tension in wire 2 is = ', str(maxT2), ('N'))
        print('The max tension occurs at the position = ', str(Max_position2), 'm')
        print()
        print('Max tension in wire 3 is = ', str(maxT3), ('N'))
        print('The max tension occurs at the position = ', str(Max_position3), 'm')
            

    elif MyInput!='q':
        print(MyInput,' is not a valid choice please try again')
        
print('Program enden \nGoodbye')


    
          
    
#Matrix=np.array([[1.0,0.0,0.0,1.0],[0.0,2.0,1.0,2.0],[2.0,1.0,0.0,1.0],[2.0,0.0,1.0,4.0]])

#print(Max_error(Invert,Matrix))



        

    
