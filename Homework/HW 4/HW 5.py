import numpy as np
import scipy
import matplotlib.pyplot as plt
import math

def driver():
    Ti = 20
    Ts = -15
    alpha = .138e-6
    f = lambda x: (Ti - Ts) * scipy.special.erf(x/ np.sqrt(4*alpha*5184000)) + Ts
    fp = lambda x: (Ti - Ts) * (2/np.sqrt(np.pi)) * (np.exp(-x**2) / np.sqrt(4*alpha*5184000))

    f1 = lambda x: np.exp(3*x) - 27*x**6 + 27*x**4*np.exp(x) - 9*x**2*np.exp(2*x)
    f1p = lambda x: 3*np.exp(3*x) - 27*6*x**5 + 27*4*x**3*np.exp(x) + 27*x**4*np.exp(x) - 18*x**2*np.exp(2*x) - 18*x*np.exp(2*x)

    f2 = lambda x: x**6-x-1
    f2p = lambda x: 6*x**5 -1

    g = lambda x: x - 3*f1(x)/f1p(x)
    gp = lambda x: 1- 3*((x**2 - 4*x + 2)*np.exp(x) + 6*x**2) / (np.exp(x)-6*x)**2

    # x = np.linspace(0,2,100)
    # y = f(x)
    # plt.figure(1)
    # plt.plot(x,y)
    # plt.show()

    a = 0
    b=2
    tol = 1e-6
    x0 = 1

    # astar, ier = bisection(f,a,b,tol)
    # print("approx root: ", astar)
    # print("ier:", ier)

    # _,astar, ier, it = newton(f,fp,x0,tol,100)
    # print("approx root: ", astar)
    # print("ier:", ier)

    a,astar, ier, it = newton(f2,f2p,3,tol,100)
    print("approx root: ", astar)
    print("ier:", ier)
    print(it)

    errors = abs(a[0:it] - astar)
    errors1 = abs(a[1:it] - astar)
    errors = np.log(errors)
    errors1 = np.log(errors1)

    print(errors)

    plt.figure(1)
    plt.plot(errors[0:-1],errors1)
    plt.xlabel("x(n+1)")
    plt.ylabel("x(n)")
    plt.show()


    return
def newton(f,fp,p0,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1);
  p[0] = p0
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,it]


def bisection(f,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier]
def secant(f,p0,p1,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1)
  p[0] = p0
  p[1] = p1
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,it]

driver()