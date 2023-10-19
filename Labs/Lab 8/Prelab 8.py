import numpy as np
import matplotlib.pyplot as plt

def driver():
    xeval = np.linspace(0,10, 1000)
    xint = np.linspace(0,10,11)

    print(np.where(np.logical_and(xeval <= xint[2], xeval >= xint[1])))

    return

def evalLineInterp(x0, f0, x1, f1, x):
    slope = (f1-f0)/(x1-x0)
    interp = slope * (x - x1) + f1
    return interp

def findIntervalPoints(xint, xeval, interval):
    ind = np.where(np.logical_and(xeval <= xint[interval+1], xeval >= xint[interval]))

    
    return ind


driver()