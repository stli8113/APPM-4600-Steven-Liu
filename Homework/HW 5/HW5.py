import numpy as np
import matplotlib.pyplot as plt

def driver():
    f = lambda x,y: 3*x**2 - y**2
    g = lambda x,y:  3*x*y**2 - x**3 - 1
    fx = lambda x,y: 6*x
    fy = lambda x,y: -2*y
    gx = lambda x,y: 3*y**2 - 3*x**2
    gy = lambda x,y: 6*x*y 

    J = np.array([[1/6, 1/18],[0,1/6]])
    
    Nmax = 100
    tol = 1e-15

    x0 = 1
    y0 = 1

    points, xstar, ystar, ier = LazyNewton2d(f,g,J,x0,y0, Nmax, tol)
    print("Root found at: ", [xstar, ystar])
    print("Error message:", ier)

    norms = np.linalg.norm(points,axis=1)
    errors = abs(norms - norms[-1])

    print(np.log(errors[-3]) / np.log(errors[-4]))

    points, xstar, ystar, ier = newton2d(f,g,fx,fy,gx,gy,x0,y0, Nmax, tol)
    print("Root found at: ", [xstar, ystar])
    print("Error message:", ier)
    
    norms = np.linalg.norm(points,axis=1)
    errors = abs(norms - norms[-1])

    print(np.log(errors[-2]) / np.log(errors[-3]))
    # print(errors)
    # plt.figure(1)
    # plt.plot(np.log(errors[0:-2]),np.log(errors[1:-1]))
    # plt.show()
    
    return


def LazyNewton2d(f,g,J, x0, y0, Nmax, tol):
    count = 0
    iterates = np.array([[x0,y0],])

    while (count < Nmax):
        x1 = x0 - J[0,0]*f(x0,y0) - J[0,1]*g(x0,y0)
        y1 = y0 - J[1,0]*f(x0,y0) - J[1,1]*g(x0,y0)
        iterates = np.append(iterates, [[x1,y1]],axis=0)
        # print(iterates)
        # print(x1,y1)
        if (abs(np.linalg.norm([x0,y0]) - np.linalg.norm([x1,y1])) < tol):
            ier = 0
            return [iterates, x1,y1, ier]
        x0 = x1
        y0 = y1

    xstar = x1
    ystar = y1
    ier = 1
    return [iterates, xstar, ystar, ier]
def newton2d(f,g,fx,fy,gx,gy,x0,y0,Nmax,tol):
    count = 0
    iterates = np.array([[x0,y0],])

    while (count < Nmax):
        J = np.array([[fx(x0,y0), fy(x0,y0)],[gx(x0,y0), gy(x0,y0)]])
        Jinv = np.linalg.inv(J)
        x1 = x0 - Jinv[0,0]*f(x0,y0) - Jinv[0,1]*g(x0,y0)
        y1 = y0 - Jinv[1,0]*f(x0,y0) - Jinv[1,1]*g(x0,y0)
        iterates = np.append(iterates, [[x1,y1]],axis=0)
        if (abs(np.linalg.norm([x0,y0]) - np.linalg.norm([x1,y1])) < tol):
            ier = 0
            return [iterates, x1,y1, ier]
        x0 = x1
        y0 = y1
    ier =1
    return [iterates, x1,y1, ier]
driver()