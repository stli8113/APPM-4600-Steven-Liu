#import libraries
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand

#part A, summation
def dotProduct(a,b,n):
    #inputs: a, b same length vectors, n length of vector
    #output: sum of element wise pruducts of a*b
    sum = 0
    for n in range(n):
        product = a[n]*b[n]
        sum += product
    return sum

def plotCircles(R, dr, f, p):
    x = lambda theta: R*(1 + dr *np.sin(f *theta + p)) * np.cos(theta)
    y = lambda theta: R*(1 + dr *np.sin(f *theta + p)) * np.sin(theta)

    thetas = np.linspace(0,2*np.pi,100)
    xval = x(thetas)
    yval = y(thetas)
    return xval, yval

#set up vectors

t = np.arange(0,np.pi, np.pi/30)
y = np.cos(t)

S = dotProduct(t,y,30)
print("The sum is:", S)

x, y = plotCircles(1.2,.1,15,0)
plt.figure(1)
plt.plot(x,y)

plt.figure(2)
for i in range(10):
    p = rand.uniform(0,2)
    x,y = plotCircles(i,.05,2+i,p)
    plt.plot(x,y)
plt.show()
