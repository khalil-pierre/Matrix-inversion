# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 04:00:06 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sp

def LU_lin_solver(A,b):
    '''Calculates the soultion to a linear equation using LU decomposition.'''
    lu,piv=sp.lu_factor(A)
    
    return sp.lu_solve((lu,piv),b)


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




b=np.array([0,0,70*9.81])


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
    
    
    

