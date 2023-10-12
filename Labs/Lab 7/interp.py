import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():


    f = lambda x: 1/  (1 + (10*x)**2)

    N = 4
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
    yeval_l= np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)
    yeval_mon = np.zeros(Neval+1)
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y = np.zeros( (N+1, N+1) )
     
    for j in range(N+1):
       y[j][0]  = yint[j]

    y = dividedDiffTable(xint, y, N+1)
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
       yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)
       yeval_mon[kk] = monomial(xeval[kk],xint,yint,N)
          

    


    ''' create vector with exact values'''
    fex = f(xeval)
       

    plt.figure(1)    
    plt.plot(xeval,fex,'ro-',label="function")
    plt.plot(xeval,yeval_l,'bs--',label="lagrange") 
    plt.plot(xeval,yeval_dd,'c.--',label="newtown dd")
    plt.plot(xeval,yeval_mon,'k.-',label="monomial")
    plt.title("Approximation for N=5 with cos distribution")
    plt.legend()

    plt.figure(2) 
    err_l = abs(yeval_l-fex)
    err_dd = abs(yeval_dd-fex)
    err_mon = abs(yeval_mon - fex)
    plt.semilogy(xeval,err_l,'ro--',label='lagrange')
    plt.semilogy(xeval,err_dd,'bs--',label='Newton DD')
    plt.semilogy(xeval,err_mon,"k.-",label="monomial")

    plt.title("Errors for N=5 with cos distribution")
    plt.legend()
    plt.show()

def monomial(xeval, xint, yint, N):
    V = np.ones((N+1,N+1))
    i = 1
    while (i < N+1):
        for j in range(N+1):
            V[j,i] = xint[j]**i
        i += 1
    Vinv = la.inv(V)
    A = np.matmul(Vinv, np.transpose(yint))
    y = 0
    for i in range(N+1):
        y += A[i] * xeval**i
    return y
    

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
  
    


''' create divided difference matrix'''
def dividedDiffTable(x, y, n):
 
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                                     (x[j] - x[i + j]))
    return y
    
def evalDDpoly(xval, xint,y,N):
    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N+1)
    
    ptmp[0] = 1.
    for j in range(N):
      ptmp[j+1] = ptmp[j]*(xval-xint[j])
     
    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
       yeval = yeval + y[0][j]*ptmp[j]  

    return yeval

       

driver()        
