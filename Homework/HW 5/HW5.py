import numpy as np
import matplotlib.pyplot as plt

def driver():
    f = lambda x,y: 3*x**2 - y**2
    g = lambda x,y:  3*x*y**2 - x**3 - 1
    fx = lambda x,y: 6*x
    fy = lambda x,y: -2*y
    gx = lambda x,y: 3*y**2 - 3*x**2
    gy = lambda x,y: 6*x*y 

    f1 = lambda x,y,z: x**2 + 4*y**2 + 4*z**2 - 16
    fx1 = lambda x,y,z: 2*x
    fy1 = lambda x,y,z: 8*y
    fz1 = lambda x,y,z: 8*z

    J = np.array([[1/6, 1/18],[0,1/6]])
    
    Nmax = 100
    tol = 1e-15

    x0 = 1
    y0 = 1
    z0 = 1

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

    points, xstar, ystar, zstar, ier = gradient3d(x0,y0,z0,f1,fx1,fy1,fz1,tol,Nmax)
    print("Root found at: ", [xstar, ystar, zstar])
    print("Error message:", ier)
    print("Root test returns:", f1(xstar, ystar, zstar))
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
def gradient3d(x0,y0,z0,f,fx,fy,fz, tol, Nmax):
    count = 0
    iterates = np.array([[x0,y0,z0],])
    while (count < Nmax):
        fatx = f(x0,y0,z0)
        fxatx = fx(x0,y0,z0)
        fyatx = fy(x0,y0,z0)
        fzatx = fz(x0,y0,z0)
        d = fatx/(fxatx**2 + fyatx**2 + fzatx**2)

        x1 = x0 - fxatx*d
        y1 = y0 - fyatx*d
        z1 = z0 - fzatx*d
        iterates = np.append(iterates, [[x1,y1,z1]],axis=0)

        if (abs(np.linalg.norm([x0,y0,z0]) - np.linalg.norm([x1,y1,z1])) < tol):
            ier = 0
            return [iterates, x1,y1,z1, ier]
        x0 = x1
        y0 = y1
        z0 = z1
    ier = 1
    return [iterates, x1,y1,z1, ier]
driver()