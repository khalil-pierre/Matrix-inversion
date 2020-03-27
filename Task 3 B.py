# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:19:20 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sp


def wire_vector(R_drum,r):
    '''Calculates the position vector that run along a wire. Uses the position of the drums and the position of the artist.'''
    return R_drum-r

def cosz(r):
    '''Calculates the cosine of the angle between the x-y plane and the wire vector.'''
    return np.sqrt((r[0]**2)+(r[1]**2))/np.linalg.norm(r)

def sinz(r):
    '''Calculates the sine of the angle between the x-y plane and the wire vector.'''
    return r[2]/np.linalg.norm(r)
    
def cosx(r):
    '''Calculates the cosine of the angle between the component of the wire vector in the x-y plane and the y axis.'''
    return abs(r[1])/np.linalg.norm(np.array([r[0],r[1]]))

def sinx(r):
    '''Calculates the sine of the angle between the component of the wire vector in the x-y plane and the y axis.'''
    return r[0]/np.linalg.norm(np.array([r[0],r[1]]))

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

def LU_lin_solver(A,b):
    '''Calculates the soultion to a linear equation using LU decomposition.'''
    lu,piv=sp.lu_factor(A)
    
    return sp.lu_solve((lu,piv),b)


CoefficientMatrix=np.zeros((3,3))
b=np.array([0,0,70*9.81])

print('For the 3D case we produce a 4D data set unfortunately we cannot represent the tension'
      ' at each position for each wire on a single graph. Instead a cross section in the x-y'
      ' plane and z-x plane will be produced for constant z and y respectivly.')

z=float(input('Please select the z position of the trapeze artist: '))

Tension_wire_1=[]
Tension_wire_2=[]
Tension_wire_3=[]

List1_3D=[]
List2_3D=[]
List3_3D=[]

R_drum1=np.array([0,0,8])
R_drum2=np.array([15,0,8])
R_drum3=np.array([15/2,8,8])


'''
for y in np.linspace(1e-6,7,100):
    for x in np.linspace(1e-6,15,100):
        if x<=7.5:
            if y<(8/7.5)*x:
                position=np.array([x,y,z])
                
                Wire_vector1=wire_vector(R_drum1,position)
                Wire_vector2=wire_vector(R_drum2,position)
                Wire_vector3=wire_vector(R_drum3,position)
                
                sin1z=sinz(Wire_vector1)
                cos1z=cosz(Wire_vector1)
                sin1x=sinx(Wire_vector1)
                cos1x=cosx(Wire_vector1)
                
                sin2z=sinz(Wire_vector2)
                cos2z=cosz(Wire_vector2)
                sin2x=sinx(Wire_vector2)
                cos2x=cosx(Wire_vector2)
                
                sin3z=sinz(Wire_vector3)
                cos3z=cosz(Wire_vector3)
                sin3x=sinx(Wire_vector3)
                cos3x=cosx(Wire_vector3)
                
                CoefficientMatrix[0]=np.array([-cos1z*sin1x,cos2z*sin2x,cos3z*sin3x])
                CoefficientMatrix[1]=np.array([-cos1z*cos1x,-cos2z*cos2x,cos3z*cos3x])
                CoefficientMatrix[2]=np.array([sin1z,sin2z,sin3z])
                
                Tension3D=LU_lin_solver(CoefficientMatrix,b)
            
                List1_3D.append(Tension3D[0])
                List2_3D.append(Tension3D[1])
                List3_3D.append(Tension3D[2])
                    
            else:
                List1_3D.append(0.0)
                List2_3D.append(0.0)
                List3_3D.append(0.0)
            
        elif x>7.5:
            if y<(-8/7.5)*x+16:
                position=np.array([x,y,z])
                
                Wire_vector1=wire_vector(R_drum1,position)
                Wire_vector2=wire_vector(R_drum2,position)
                Wire_vector3=wire_vector(R_drum3,position)
                
                sin1z=sinz(Wire_vector1)
                cos1z=cosz(Wire_vector1)
                sin1x=sinx(Wire_vector1)
                cos1x=cosx(Wire_vector1)
                
                sin2z=sinz(Wire_vector2)
                cos2z=cosz(Wire_vector2)
                sin2x=sinx(Wire_vector2)
                cos2x=cosx(Wire_vector2)
                
                sin3z=sinz(Wire_vector3)
                cos3z=cosz(Wire_vector3)
                sin3x=sinx(Wire_vector3)
                cos3x=cosx(Wire_vector3)
                
                CoefficientMatrix[0]=np.array([-cos1z*sin1x,cos2z*sin2x,-cos3z*sin3x])
                CoefficientMatrix[1]=np.array([-cos1z*cos1x,-cos2z*cos2x,cos3z*cos3x])
                CoefficientMatrix[2]=np.array([sin1z,sin2z,sin3z])
                
                Tension3D=LU_lin_solver(CoefficientMatrix,b)
            
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

plt.imshow(Tension_wire_1)
   
'''
 
for y in np.linspace(1e-6,7,100):
    for x in np.linspace(1e-6,15,100):
        if y<(8/7.5)*x and y<(-8/7.5)*x+16:
            
            position=np.array([x,y,z])
            
            Wire_vector1=wire_vector(R_drum1,position)
            Wire_vector2=wire_vector(R_drum2,position)
            Wire_vector3=wire_vector(R_drum3,position)
            
            sin1z=sinz(Wire_vector1)
            cos1z=cosz(Wire_vector1)
            sin1x=sinx(Wire_vector1)
            cos1x=cosx(Wire_vector1)
            
            sin2z=sinz(Wire_vector2)
            cos2z=cosz(Wire_vector2)
            sin2x=sinx(Wire_vector2)
            cos2x=cosx(Wire_vector2)
            
            sin3z=sinz(Wire_vector3)
            cos3z=cosz(Wire_vector3)
            sin3x=sinx(Wire_vector3)
            cos3x=cosx(Wire_vector3)
            
            if x>7.5:
                
                CoefficientMatrix[0]=np.array([-cos1z*sin1x,cos2z*sin2x,cos3z*sin3x])
                CoefficientMatrix[1]=np.array([-cos1z*cos1x,-cos2z*cos2x,cos3z*cos3x])
                CoefficientMatrix[2]=np.array([sin1z,sin2z,sin3z])

            if x<7.5:
                
                CoefficientMatrix[0]=np.array([-cos1z*sin1x,cos2z*sin2x,-cos3z*sin3x])
                CoefficientMatrix[1]=np.array([-cos1z*cos1x,-cos2z*cos2x,cos3z*cos3x])
                CoefficientMatrix[2]=np.array([sin1z,sin2z,sin3z])
            
            Tension3D=LU_lin_solver(CoefficientMatrix,b)
            
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
plt.imshow(Tension_wire_1,origin='lower')
plt.colorbar()
plt.show()
plt.clf()

plt.title('Tension in wire 2 in the x-y plane')      
plt.imshow(Tension_wire_2,origin='lower')
plt.colorbar()
plt.show()
plt.clf()

plt.title('Tension in wire 3 in the x-y plane')      
plt.imshow(Tension_wire_3,origin='lower')
plt.colorbar()
plt.show()
plt.clf()