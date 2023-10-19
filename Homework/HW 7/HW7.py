import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def driver():
    f = lambda x: 1/  (1 + (10*x)**2)

    N = 8
    ''' interval'''
    a = -1
    b = 1
   
   
    ''' create equispaced interpolation nodes'''
    xint = np.linspace(a,b,N+1)
    for i in range(N+1):
        xint[i] = np.cos(((2*(i+1) - 1)*np.pi) / (2*(N+1)))

    ''' create interpolation data'''
    yint = f(xint)
    # print(xint, yint)

    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_bary= np.zeros(Neval+1)
    yeval_l= np.zeros(Neval+1)

    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_bary[kk] = eval_bary(xeval[kk],xint,yint,N)   
       yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)


    ''' create vector with exact values'''
    fex = f(xeval)

    print(yeval_bary[0], yeval_l[0])

    plt.figure(1)    
    plt.plot(xeval,fex,'ro-',label="function")
    plt.plot(xeval,yeval_bary,'bs--',label="barycentric") 
    plt.plot(xeval,yeval_l,'ko--',label="lagrange") 
    plt.title("Approximation for N=5 with cos distribution")
    plt.legend()
    plt.show()
    return

def eval_bary(xeval,xint,yint,N):

    lj = np.ones(N+1)
    phi = eval_phi(xeval, xint)
    
    for jj in range(N+1):
        wj = eval_w_j(xint, jj)
        if(xeval != xint[jj]):
            lj[jj] = (wj / (xeval - xint[jj]))
        else:
            lj[jj] = 0


    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]

    yeval = yeval * phi
  
    return(yeval)

def eval_phi(x, xint):
    phi = 1
    for xi in xint:
        phi *= (x - xi)
    return phi

def eval_w_j(xint, j):
    wj = 1
    xj = xint[j]
    xint = np.delete(xint, j)
    for xi in xint:
        wj = wj / (xj - xi)
    return wj

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

driver()