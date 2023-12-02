import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
def driver():
    f = lambda x: 1 / (1 + x**2)
    f = lambda x: x * np.cos(1/x)
    a = 0
    b = 1

    print(simpsons(a, b, 8, f))

    # f1 = lambda x: np.sin(x)
    # McL = lambda x: abs(f1(x) - 1 - x**3/6 + x**5/120)
    # P33 = lambda x: abs(f1(x) - (x - 7*x**3/60) / (1 + x**2/20))
    # P42 = lambda x: abs(f1(x) - (x) / (1 + x**2/6 + 7*x**4/120))
    # P24 = lambda x: abs(f1(x) - (x - 7*x**3/60) / (1 + x**2/20))
    
    # xeval = np.linspace(0,5,100)
    # mcErr = McL(xeval)
    # P33Err = P33(xeval)
    # P42Err = P42(xeval)
    # P24Err= P24(xeval)

    # plt.plot(xeval, mcErr, "b", label="MacLaurin Error")
    # plt.plot(xeval, P33Err, "y", label="Cubic Error")
    # plt.plot(xeval, P42Err, "g", label="P_2^4 Error")
    # plt.plot(xeval, P24Err, "k", label="P_4^2 Error")
    # plt.title("Comparison of Approximation Errors")
    # plt.ylabel("y")
    # plt.xlabel("x")
    # plt.legend()
    # plt.show()

    # y, err, infodict= quad(f, a, b, full_output=1)
    # y, err, infodict= quad(f, a, b, full_output=1, epsabs=10**-4)
    # print(infodict["neval"])
    # print(y)
    # print(trapz(a,b,1291, f))
    # print(simpsons(a,b,108,f))
    return


def trapz(a, b, n, f):
    h = (b - a)/(n)
    xeval = np.linspace(a, b, n+1)
    yeval = f(xeval)
    eval_sum = yeval[0] + yeval[-1] + 2*np.sum(yeval[1:-1])
    eval_sum = (h/2) * eval_sum

    return eval_sum

def simpsons(a, b, n, f):
    if(n % 2 != 0):
        print("N is not even!")
        return 0
    
    h = (b-a)/(n)
    xeval = np.linspace(a,b,n+1)
    yeval = np.zeros(n+1)
    yeval[1:] = f(xeval[1:])
    # print(yeval)
    eval_sum = yeval[0] + yeval[-1] + 4 * np.sum(yeval[1:-1:2]) + 2 * np.sum(yeval[2:-1:2])
    # print(yeval[1:-1:2])
    # print(4 * np.sum(yeval[1:-1:2]))
    eval_sum = (h/3) * eval_sum

    return eval_sum
    


driver()