import numpy as np
import matplotlib.pyplot as plt

def driver():
    f1 = lambda x: 5*x/4 - np.sin(2*x) - 3/4

    x = np.linspace(-2,6,100)
    y = f1(x)
    
    x0 = 4.5
    tol = 10e-10
    Nmax = 100
    [astar, ier] = fixedpt(f1, x0, tol, Nmax)
    print("root is:", astar)
    print("error code is:", ier)
    print(f1(astar))

    '''
    plt.figure(1)
    plt.plot(x,y)
    plt.plot(x, np.zeros(len(x)))
    plt.title("Problem 5 Plot")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()
    '''


def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [xstar,ier]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]
driver()