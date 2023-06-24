import numpy as np 
from scipy.integrate import solve_bvp

def fun(x, y):
    return np.array([y[1], -np.exp(y[0])])

def bc(ya, yb):
    return np.array([ya[0], yb[0]])

x_span = [0, 1]
N = 5
x = np.linspace(x_span[0], x_span[1], N)

y_a = np.zeros((2, x.size))
y_b = np.zeros((2, x.size))
y_b[0] = 3

res_a = solve_bvp(fun, bc, x, y_a)
res_b = solve_bvp(fun, bc, x, y_b)

print(res_a)
print(res_b)

import matplotlib.pyplot as plt 

x_plot = np.linspace(x_span[0], x_span[1], 1000)
y_plot_a = res_a.sol(x_plot)[0]
y_plot_b = res_b.sol(x_plot)[0]

plt.figure()
plt.plot(x_plot, y_plot_a, label='y_a')
plt.plot(x_plot, y_plot_b, label='y_b')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('solve_bvp_ex5.png')