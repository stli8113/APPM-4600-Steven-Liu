# import libraries
import numpy as np
    
def driver():

# test functions 
     f1 = lambda x: (10/(x+4))**0.5

     Nmax = 100
     tol = 1e-10

# test f1 '''
     x0 = 1.5
     [xstars,ier] = fixedpt(f1,x0,tol,Nmax)
     print(xstars)
     print('the approximate fixed point is:',xstars[-1])
     print('f1(xstar):',f1(xstars[-1]))
     print('Error message reads:',ier)

     p_n = aitken(xstars,tol,len(xstars))
     print("Aitken converges in", len(p_n), "iterations")
     print("Aitken converges to:", p_n[-1])

     stefEst = steffenson(x0,f1,Nmax,tol)
     print("Steffenson converges to:",stefEst)

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

def aitken(p, tol, Nmax):
   ''' p = array of fxpoint sequence''' 
   ''' Nmax = max number of iterations'''
   ''' tol = stopping tolerance'''
   p_n = np.zeros(Nmax) 
   count = 0
   while count + 3 <= Nmax:
      p_n[count] = p[count] - (p[count+1] - p[count])**2 / (p[count+2] - 2*p[count+1] + p[count])
      if (abs(p_n[count] - p_n[count-1]) < tol):
         return p_n[0:count-2]
      count += 1
   return p_n[0:count-2]

def steffenson(x0, f1, Nmax, tol):
   ''' x0 = initial guess''' 
   ''' Nmax = max number of iterations'''
   ''' tol = stopping tolerance'''
   x1 = f1(x0)
   x2 = f1(x1)
   count = 0
   while (count <= Nmax):
      xEst = x0 - (x1 - x0)**2 / (x2 - 2*x1 + x0)
      if abs(xEst - x0) < tol:
         return xEst
      x0 = xEst
      x1 = f1(xEst)
      x2 = f1(x1)
      
      count += 1


driver()