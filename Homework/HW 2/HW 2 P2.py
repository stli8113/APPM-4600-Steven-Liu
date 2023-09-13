import numpy as np
import numpy.linalg as la

A = 0.5 * np.array([[1,1],[1+10**(-10), 1-10**(-10)]])

U, S, Vh = la.svd(A)

print(S)
print(U)