import numpy as np

#-------------------------------
x0 = 0
xN = np.pi/2

N = 5
h = (xN - x0)/N 

x = np.arange(x0, xN+h, h)

A = np.eye(N+1, k=1) + np.eye(N+1, k=-1) + (-2 + 4*(h**2))*np.eye(N+1, k=0)
A[0, :] = np.concatenate(([1], np.zeros(N)))
A[-1, -2] = 2
print(A)

b = np.zeros(N+1)
b[1:] = (4*h**2)*x[1:]
print(b)

y = np.linalg.solve(A, b)

#-------------------------------
from scipy.integrate import solve_bvp
def F(x, y):
    return np.array([y[1], -4*y[0]+4*x])

x_span = [0, np.pi/2]
x_eval = np.arange(x_span[0], x_span[1]+h, h)

def BC(ya, yb):
    return np.array([ya[0], yb[1]])

y0 = np.zeros((2, len(x_eval)))

sol = solve_bvp(F, BC, x_eval, y0)
print(sol)

import matplotlib.pyplot as plt 
plt.figure()
plt.plot(x, y, label='Finite Difference')

f_exact = lambda x: x + 0.5*np.sin(2*x)
y_exact = f_exact(x)
plt.plot(x, y_exact, '--', label='exact')

plt.plot(sol.x, sol.y[0], ':', label='solve_bvp')

plt.legend()
plt.savefig('solve_bvp_ex4.png')