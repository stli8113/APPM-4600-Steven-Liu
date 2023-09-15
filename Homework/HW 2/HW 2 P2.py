import numpy as np
import numpy.linalg as la
import math

#A = 0.5 * np.array([[1,1],[1+10**(-10), 1-10**(-10)]]) 

A = [[.5,.5],[.5*(1+10**(-10)), .5*(1-10**(-10))]]
invA = [[1-10**10, 10**10],[1+10**10,-10**10]]

print(la.cond(A))

b = np.array([[1],[1]])
db = 10**-5 * np.array([[1],[1]])


x = np.matmul(invA,b)
dx = np.matmul(invA, db)

print(x, "\n", dx)

print("the relative error is:",la.norm(dx)/la.norm(x))


R = lambda n,x: np.exp(x)/math.factorial(n+1) * x**(n+1) 

y = lambda x: np.exp(x) - 1
P = lambda x:  x + .5 * x**2

x = 9.999999995000000 * 10**-10
realY = 10**-9

#print(R(2,x))
print(P(x))
#print(y(x), realY)