import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
from scipy.integrate import quad

def driver():
    print("starting\n")
    # function you want to approximate
    f = lambda x: 1 / (1 + x**2)
    # interval of interest
    a = -1
    b = 1
    # weight function
    w = lambda x: 1.
    # order of approximation
    order = 2
    # number of points you want to sample in [a,b]
    n = 1000
    xeval = np.linspace(a,b,n+1)
    pval = np.zeros(n+1)
    for kk in range(n+1):
        # print(kk)
        pval[kk] = eval_legendre_expansion(f,a,b,w,order,xeval[kk])
    
    '''create vector with exact values'''
    fex = np.zeros(n+1)
    for kk in range(n+1):
        # print(kk)
        fex[kk] = f(xeval[kk])
    
    plt.figure(1)
    plt.plot(xeval,fex,"ro-", label= "f(x)")
    plt.plot(xeval,pval,"bs--",label= "expansion")
    plt.legend()

    plt.figure(2)
    err = abs(pval-fex)
    plt.semilogy(xeval,err,"ro--",label="error")
    plt.legend()
    plt.show()

def eval_legendre(xeval, n):
    p = np.zeros(n+1)
    p[0] = 1
    p[1] = xeval
    count = 1
    while(count < n):
        phi = lambda x: 1/(count+1) * ((2*count+1)*x*p[count] - count * p[count-1])
        p[count+1] = phi(xeval)
        count += 1

    return p

def eval_legendre_expansion(f,a,b,w,n,xeval):
    # this subroutine evaluates the legendre expansion
    # evaluate all the legendre polynomials at x that are needed
    # by calling your code from prelab
    p = eval_legendre(xeval, n)
    # print(p)
    # initialize the sum to 0
    pval = 0.0
    for j in range(0,n+1):
        # make a function handle for evaluating phi_j(x)
        phi_j = lambda x: eval_legendre(x, n)[j]
        # make a function handle for evaluating phi_j^2(x)*w(x)
        phi_j_sq = lambda x: phi_j(x) ** 2 * w(x)
        # use the quad function from scipy to evaluate normalizations
        norm_fac,err = quad(phi_j_sq, a, b)
        # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
        func_j = lambda x: phi_j(x) * f(x) * w(x) / norm_fac
        # use the quad function from scipy to evaluate coeffs
        aj,err = quad(func_j, a, b)
        # accumulate into pval
        pval = pval+aj*p[j]
    return pval

driver()