import numpy as np
import scipy

def evaluateGamma(x):
    f = lambda t: t**(x-1) * np.exp(-t)

    cutoff = 5*x 
    n = 25*x
    gamma = trapz(0,cutoff,n,f)

    return gamma

def evalGauss(n, f, x):
    points, weights = np.polynomial.laguerre.laggauss(n)
    # print(points, weights)
    sum = 0
    
    for i in range(len(points)):
        # print(f(x,points[i]))
        sum = sum + f(x, points[i])*weights[i]
    # print("\n ", sum)
    return sum

def trapz(a, b, n, f):
    h = (b - a)/(n)
    xeval = np.linspace(a, b, n+1)
    yeval = f(xeval)
    eval_sum = yeval[0] + yeval[-1] + 2*np.sum(yeval[1:-1])
    eval_sum = (h/2) * eval_sum

    return eval_sum

def driver():
    f = lambda x, t: t**(x-1)


    values = [2,4,6,8,10]
    gammas = np.zeros(5)
    scipyGammas = np.zeros(5)
    gaussGammas = np.zeros(5)
    for i in range(5):
        scipyGammas[i] = scipy.special.gamma(values[i])
        gammas[i] = evaluateGamma(values[i])
        gaussGammas[i] = evalGauss(5, f, values[i])

    print(gaussGammas)

driver()
