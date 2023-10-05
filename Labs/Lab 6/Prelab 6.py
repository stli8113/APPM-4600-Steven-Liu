import numpy as np
import matplotlib.pyplot as plt

def driver():
    f = lambda x: np.cos(x)
    forwardDiff = lambda s,h: (f(s+h) - f(s))/h
    centerDiff = lambda s,h: (f(s+h) - f(s-h))/(2*h)
    
    h = 0.01 * 2. **(-np.arange(0, 10))
    cent = centerDiff(np.pi/2,h)
    forward = forwardDiff(np.pi/2,h)

    print(cent)
    print(forward)

driver()
