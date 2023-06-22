import numpy as np
from scipy.integrate import solve_bvp

def fun(x, y, p):
    k = p[0]
    return np.array([y[1], -k**2 * y[0]])

def bc(ya, yb, p):
    k = p[0]
    return np.array([ya[0], ya[1]-k, yb[0]])

x = np.linspace(0, 1, 5)
y0 = np.zeros((2, x.size))
y0[0, 1] = 1
y0[0, 3] = -1 
p0 = [6]

sol = solve_bvp(fun, bc, x, y0, p=p0)
print(sol)

import matplotlib.pyplot as plt 
x_plot = np.linspace(0, 1, 1000)
y_plot = sol.sol(x_plot)[0]

plt.figure()
plt.plot(x_plot, y_plot)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('solve_bvp_ex6.png')