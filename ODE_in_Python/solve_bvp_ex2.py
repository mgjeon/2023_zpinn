import numpy as np 
from scipy.integrate import solve_ivp

def F(t, y):
    g = 9.8
    return np.array([y[1], -g])

t_span = [0, 5]
h = 0.1
t = np.arange(t_span[0], t_span[1]+h, h)

from scipy.optimize import fsolve

def f(x):
    sol = solve_ivp(F, t_span, [0, x[0]], t_eval=t)
    return sol.y[0][-1] - 50

v0, = fsolve(f, [10])
print(v0)

sol = solve_ivp(F, t_span, [0, v0], t_eval=t)
print(sol)

import matplotlib.pyplot as plt 

plt.figure()
plt.plot(5, 50, 'ro')
plt.plot(0, 0, 'ro')

plt.plot(sol.t, sol.y[0])
plt.xlabel('t')
plt.ylabel('y(t)')

plt.savefig('solve_bvp_ex2.png')
