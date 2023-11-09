import numpy as np
from scipy.integrate import quad
def driver():
    f = lambda x: 1 / (1 + x**2)
    a = -5
    b = 5

    y, err, infodict= quad(f, a, b, full_output=1)
    # y, err, infodict, _= quad(f, a, b, full_output=1, epsabs=10**-4)
    print(infodict["neval"])
    print(y)
    print(trapz(a,b,1291, f))
    print(simpsons(a,b,108,f))
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
    yeval = f(xeval)
    # print(yeval)
    eval_sum = yeval[0] + yeval[-1] + 4 * np.sum(yeval[1:-1:2]) + 2 * np.sum(yeval[2:-1:2])
    # print(yeval[1:-1:2])
    # print(4 * np.sum(yeval[1:-1:2]))
    eval_sum = (h/3) * eval_sum

    return eval_sum
    


driver()