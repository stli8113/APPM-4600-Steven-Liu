import numpy as np
import math
import matplotlib.pyplot as plt

#initialize deltas
delta = np.logspace(-16,0,17) 

smallX = np.pi
bigX = 10**5

def badCosine(x,delta):
    y = np.cos(x+delta) - np.cos(x)
    return y

def goodCosine(x,delta):
    y = -2*np.sin((2*x + delta)/2)*np.sin(delta/2)
    return y

def taylorCosine(x,delta):
    y = -delta*np.sin(x) - delta**2/2 * np.cos(x)
    return y


bigDiff = badCosine(bigX,delta) - goodCosine(bigX,delta)
smallDiff = badCosine(smallX,delta) - goodCosine(smallX,delta)

print(taylorCosine(bigX,delta))

bigDiffTay = badCosine(bigX,delta) - taylorCosine(bigX,delta)
smallDiffTay = badCosine(smallX,delta) - taylorCosine(smallX,delta)

plt.figure(1)
plt.semilogx(delta,bigDiff)
plt.title("Difference Between Functions for x = 10^5")
plt.xlabel("delta")
plt.ylabel("Difference")

plt.figure(2)
plt.semilogx(delta,smallDiff)
plt.title("Difference Between Functions for x = pi")
plt.xlabel("delta")
plt.ylabel("Difference")

plt.figure(3)
plt.loglog(delta,bigDiffTay)
plt.title("Difference Between Functions for x = 10^5 With Taylor Series")
plt.xlabel("delta")
plt.ylabel("Difference")

plt.figure(4)
plt.loglog(delta,smallDiffTay)
plt.title("Difference Between Functions for x = pi With Taylor Series")
plt.xlabel("delta")
plt.ylabel("Difference")

plt.show()