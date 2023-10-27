import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm


def driver():
    
    f = lambda x: 1 / (1 + x**2)
    fp = lambda x: -2*x/(1.+x**2)**2

    a = -5
    b = 5

    ''' number of intervals'''
    Nint = 19
    xint = np.linspace(a,b,Nint+1)
    for i in range(Nint+1):
        xint[i] = np.cos(((2*(i+1) - 1)*np.pi) / (2*(Nint+1)))
    
    xint = xint * -5
    yint = f(xint)
    ypint = fp(xint)

    ''' create points you want to evaluate at'''
    Neval = 100
    xeval =  np.linspace(xint[0],xint[Nint],Neval+1)

    
    # fp0 = fp(xint[0])
    # fpn = fp(xint[-1]) 
    (M,C,D) = create_natural_spline(yint,xint,Nint)
    (M1,C1,D1) = create_clamped_spline(yint,xint,Nint, ypint[0], ypint[-1])
    
    # print('M =', M)
#    print('C =', C)
#    print('D=', D)
    
    yeval = eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D)
    yeval_clamp = eval_cubic_spline(xeval,Neval,xint,Nint,M1,C1,D1)
    yevalL = np.zeros(Neval+1)
    yevalH = np.zeros(Neval+1)
    for kk in range(Neval+1):
        yevalL[kk] = eval_lagrange(xeval[kk],xint,yint,Nint)
        yevalH[kk] = eval_hermite(xeval[kk],xint,yint,ypint,Nint)

    # print('yeval = ', yeval)
    ''' evaluate f at the evaluation points'''
    fex = f(xeval)
        
    nerr = norm(fex-yeval)
    nerrH = norm(fex - yevalH)
    nerrL = norm(fex - yevalL)
    nerrC = norm(fex - yeval_clamp)
    print('nerr = ', nerr)
    print('nerrH = ', nerrH)
    print('nerrL = ', nerrL)
    print('nerrC = ', nerrC)

    
    plt.figure()    
    plt.plot(xeval,fex,'ro-',label='exact function')
    plt.plot(xeval,yevalH,'ko-',label='Hermite')
    plt.plot(xeval,yevalL,'mo-',label='Lagrange')
    plt.plot(xeval,yeval,'bs--',label='natural spline') 
    plt.plot(xeval,yeval_clamp,'ys--',label='clamped spline') 
    # plt.semilogy()
    plt.legend()
     
    err = abs(yeval-fex)
    errH = abs(yevalH-fex)
    errL = abs(yevalL-fex)
    errclamp = abs(yeval_clamp-fex)
    plt.figure() 
    plt.semilogy(xeval,errclamp,'bo--',label='absolute error clamped')
    plt.semilogy(xeval,err,'ko--',label='absolute error natural')
    plt.semilogy(xeval,errL,'yo--',label='absolute error Lagrange')
    plt.semilogy(xeval,errH,'go--',label='absolute error Hermite')
    plt.legend()
    plt.show()
    
def create_natural_spline(yint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)  
    for i in range(1,N):
       hi = xint[i]-xint[i-1]
       hip = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
       h[i-1] = hi
       h[i] = hip

#  create matrix so you can solve for the M values
# This is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    A[0][0] = 1.0
    for j in range(1,N):
       A[j][j-1] = h[j-1]/6
       A[j][j] = (h[j]+h[j-1])/3 
       A[j][j+1] = h[j]/6
    A[N][N] = 1

    Ainv = inv(A)
    
    M  = Ainv.dot(b)

#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = yint[j]/h[j]-h[j]*M[j]/6
       D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
    return(M,C,D)



def create_clamped_spline(yint,xint,N, fp0, fpn):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)  
    for i in range(1,N):
       hi = xint[i]-xint[i-1]
       hip = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
       h[i-1] = hi
       h[i] = hip
    b[N] = -fpn + (yint[-1] - yint[-2]) / (xint[-1] - xint[-2])
    b[0] = -fp0 + (yint[1] - yint[0]) / (xint[1] - xint[0])

#  create matrix so you can solve for the M values
# This is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    h0 = xint[1] - xint[0]
    A[0][0] = h0 / 3
    A[0][1] = h0 / 6
    for j in range(1,N):
       A[j][j-1] = h[j-1]/6
       A[j][j] = (h[j]+h[j-1])/3 
       A[j][j+1] = h[j]/6
    hn = xint[N-1] - xint[N]
    A[N][N] = hn / 3
    A[N][N-1] = hn / 6

    Ainv = inv(A)
    
    M  = Ainv.dot(b)

#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = yint[j]/h[j]-h[j]*M[j]/6
       D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
    return(M,C,D)       
def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Mip = M_{i+1}; Mi = M_i
    # print(xip)
    # print(xi) 
    # print(xeval)
    # print(Mip)
    hi = xip-xi
    yeval = (Mi*(xip-xeval)**3 +(xeval-xi)**3*Mip)/(6*hi) \
            + C*(xip-xeval) + D*(xeval-xi)
    return yeval 
    
    
def  eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
        
#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]
        # print(xeval)
        # print(xint)
# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
        # print('yloc = ', yloc)
#   copy into yeval
        yeval[ind] = yloc

    return(yeval)
           
def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)

def eval_hermite(xeval,xint,yint,ypint,N):

    ''' Evaluate all Lagrange polynomials'''

    lj = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    ''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N+1)
#    lpj2 = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
#              lpj2[count] = lpj2[count]*(xint[count] - xint[jj])
              lpj[count] = lpj[count]+ 1./(xint[count] - xint[jj])
              

    yeval = 0.
    
    for jj in range(N+1):
       Qj = (1.-2.*(xeval-xint[jj])*lpj[jj])*lj[jj]**2
       Rj = (xeval-xint[jj])*lj[jj]**2
#       if (jj == 0):
#         print(Qj)
         
#         print(Rj)
#         print(Qj)
#         print(xeval)
 #        return
       yeval = yeval + yint[jj]*Qj+ypint[jj]*Rj
       
    return(yeval)
 
           
driver()               

