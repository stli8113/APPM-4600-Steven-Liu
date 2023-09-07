import numpy as np
import numpy.linalg as la
import math

def driver():
     n = 200

     x = np.array([1,1])
     y = np.array([1,-1])

     mx = np.random.rand(n,n)
     my = np.random.rand(n,n)

# evaluate the dot product of y and w     
     dp = dotProduct(x,y,2)
     matprod = matrixMult(mx,my,n)

# print the output
     ''
     print('the dot product is : ', dp)
     print('matrix mult = ',matprod)
     ''
     #print(np.dot(x,y))
     #print(np.matmul(mx,my))

     return
     
def dotProduct(x,y,n):

     dp = 0.
     for j in range(n):
        dp = dp + x[j]*y[j]

     return dp  

def matrixMult(x,y,n):
    product = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            product[i][j] = dotProduct(x[i],y[:,j],n)
    return product

driver()               
