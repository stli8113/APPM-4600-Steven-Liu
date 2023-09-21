# import libraries
import numpy as np
    
def driver():

# test functions 
     f1 = lambda x: x * (1 + (7-x**5)/x**2)**3
     f2 = lambda x: x - ((x**5-7)/x**2)
     f3 = lambda x: x - ((x**5-7)/(5*x**4))
     f4 = lambda x: x - ((x**5-7)/12)
     #fixed point at 7^1/5 for all f1-f4
     print(f2(1.5))

     Nmax = 100
     tol = 1e-10

# test f1 '''
     x0 = .5
     [xstars,ier] = fixedpt(f3,x0,tol,Nmax)
     print(xstars)
     print('the approximate fixed point is:',xstars[-1])
     print('f1(xstar):',f1(xstars[-1]))
     print('Error message reads:',ier)
    

# define routines
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    xstars = np.zeros(Nmax)
    xstars[0] = x0

    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstars[count] = x1
          ier = 0
          return [xstars[0:count],ier]
       x0 = x1
       xstars[count] = x1

    xstars[-1] = x1
    ier = 1
    return [xstars, ier]
    

driver()