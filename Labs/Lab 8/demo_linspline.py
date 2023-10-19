import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from scipy.sparse import diags


def driver():
    
    f = lambda x: 1/ (1 + (10*x)**2)
    a = -1
    b = 1
    
    ''' create points you want to evaluate at'''
    Neval = 100
    xeval =  np.linspace(a,b,Neval)
    
    ''' number of intervals'''
    Nint = 10
    
    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)
    
    ''' evaluate f at the evaluation points'''
    fex = np.zeros(Neval)
    for j in range(Neval):
      fex[j] = f(xeval[j]) 

    M = eval_Ms(10,f,a,b)
    print(M)
      
    
    plt.figure()
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval,'bs-')
    # plt.legend()
     
    err = abs(yeval-fex)
    plt.figure()
    plt.plot(xeval,err,'ro-')
    plt.show()
    
    
def eval_Ms(Neval, f, a, b):
   offsets = [-1,0,1]
   diagVals = [np.ones(Neval-3)/12, np.ones(Neval-2)/3, np.ones(Neval-3)/12]
   x = np.linspace(a,b,Neval+1)
   y = np.zeros(Neval-2)

   dx = x[1] - x[0]
   M = diags(diagVals, offsets).toarray()
   M = np.array(M)

   for i in range(Neval-2):
      y[i] = (f(x[i+2]) - 2*f(x[i+1]) + f(x[i]))/(2*dx**2)
   Minv = inv(M)

   print(np.shape(M), np.shape(y))
   coeffM = np.matmul(Minv, y)

   return coeffM

    
    
def  eval_lin_spline(xeval,Neval,a,b,f,Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 
    
    
    for jint in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''
        
        ind = np.where(np.logical_and(xeval <= xint[jint+1], xeval >= xint[jint]))
        n = len(ind)
        
        '''temporarily store your info for creating a line in the interval of 
         interest'''
        a1= xint[jint]
        fa1 = f(a1)
        b1 = xint[jint+1]
        fb1 = f(b1)
        
        for kk in range(n):
           '''use your line evaluator to evaluate the lines at each of the points 
           in the interval'''
           '''yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with 
           the points (a1,fa1) and (b1,fb1)'''
           slope = (fb1-fa1)/(b1-a1)
           interp = slope * (xeval[ind[kk]] - a1) + fa1
           yeval[ind[kk]] = interp
    # print(yeval)
    return yeval

def  eval_poly_spline(xeval,Neval,a,b,f,Nint, M):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 
    
    
    for jint in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''
        
        ind = np.where(np.logical_and(xeval <= xint[jint+1], xeval >= xint[jint]))
        n = len(ind)
        
        '''temporarily store your info for creating a line in the interval of 
         interest'''
        a1= xint[jint]
        fa1 = f(a1)
        b1 = xint[jint+1]
        fb1 = f(b1)
        h = b1-a1
        
        for kk in range(n):
           '''use your line evaluator to evaluate the lines at each of the points 
           in the interval'''
           '''yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with 
           the points (a1,fa1) and (b1,fb1)'''
           C = fa1/h - h/6 * M[kk]
    # print(yeval)
    return yeval
           
# if __name__ == '__main__':
#       # run the drivers only if this is called from the command line
driver()               
