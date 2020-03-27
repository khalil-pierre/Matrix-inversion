import numpy as np

def Gauss_Elimination(A):
    #Calculates the inverse of a matrix using gauss jordan elimination
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
            
M=np.array([[1,1,1],[1,2,-1],[2,3,1]])
print(Gauss_Elimination(M))