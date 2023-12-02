import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from scipy.sparse import diags

def sampleFunctionWithNoise(f, N, sigma, a, b):
    xeval = np.linspace(a, b, N)
    yeval = f(xeval) + rand.normal(scale=sigma, size=N)
    return yeval

def generateTikhonovRegression(yeval, xeval, order, tikMatrix, scaling):
    A = np.ones((len(xeval), order))

    # print(xeval)
    for i in range(len(xeval)):
        for j in range(order):
            # print(i, j)
            A[i,j] = xeval[i]**j
    Atranspose = np.transpose(A)
    tikTranspose = np.transpose(tikMatrix)
    regressionCoeff = np.matmul(np.linalg.inv((np.matmul(Atranspose,A) + scaling**2 * np.matmul(tikTranspose, tikMatrix))), np.matmul(Atranspose, yeval))
    return regressionCoeff

def evalPolyRegression(xeval, coeff):
    y = np.zeros(len(xeval))
    for i in range(len(coeff)):
        orderTerm = xeval**i * coeff[i]
        y += orderTerm
    return y

def generateLeastSquares(yeval, xeval, order):
    A = np.ones((len(xeval), order))

    # print(xeval)
    for i in range(len(xeval)):
        for j in range(order):
            # print(i, j)
            A[i,j] = xeval[i]**j 
    
    Atranspose = np.transpose(A)
    regressionCoeff = np.matmul(np.linalg.inv(np.matmul(Atranspose, A)), np.matmul(Atranspose, yeval))
    return regressionCoeff


def driver():
    f = lambda x: np.sin(x) + np.sin(5*x)
    a = 1
    b = 2
    N = 50
    sigma = .5

    xeval = np.linspace(a,b,N)
    xsample = np.linspace(a,b,1000)
    yeval = sampleFunctionWithNoise(f, N, sigma, a, b)


    D = diags([-0.5, 0, 0.5], [0, 1, 2], shape=(5, 5)).toarray()
    coeff = generateTikhonovRegression(yeval, xeval, 5, D, 1)
    coeffLS = generateLeastSquares(yeval, xeval, 5)
    # print(coeff)
    # regress = lambda x: coeff[0] + coeff[1]*x + coeff[2]*x**2

    plt.figure(1)
    plt.plot(xeval, yeval, label="Sampled Function")
    plt.plot(xsample,f(xsample), label="Exact Function")
    plt.plot(xsample, evalPolyRegression(xsample, coeff), label="TLS")
    plt.plot(xsample, evalPolyRegression(xsample, coeffLS), label="LS")
    plt.legend()

    plt.figure(2)
    plt.plot(xsample, abs(f(xsample) - evalPolyRegression(xsample, coeff)), label="TLS")
    plt.plot(xsample, abs(f(xsample) - evalPolyRegression(xsample, coeffLS)), label="LS")
    plt.legend()

    plt.show()

driver()