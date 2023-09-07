import numpy as np
import matplotlib.pyplot as plt

#define anonymous functions for coefficient and exponent forms
x1 = lambda x: x**9 - 18*x**8 +144*x**7 - 672*x**6 + 2016*x**5 - 4032*x**4 + 5376*x**3 - 4608*x**2 +2304*x - 512
x2 = lambda x: (x - 2)**9

#define x input and evaluate function outputs
x = np.arange(1.92,2.08,0.001)
coeffOutput = x1(x)
expOutput = x2(x)


#plot figures
plt.figure(1)
plt.title("Output With Coefficients")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,coeffOutput)

plt.figure(2)
plt.title("Output as Single Exponent Expression")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,expOutput)

plt.show()