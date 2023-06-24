import numpy as np
from scipy.integrate import solve_ivp 

# def F(t, y):
#     return np.array([t**2*y[1], -t*y[0]])

def F(t, y):
    return np.dot(np.array([[0, t**2], [-t, 0]]), y)

t_span = [0, 10]
h = 0.01
t = np.arange(t_span[0], t_span[1]+h, h)

S0 = np.array([1, 1])

sol = solve_ivp(F, t_span, S0, t_eval=t)
print(sol)

import matplotlib.pyplot as plt 

plt.figure(figsize=(6, 10))

plt.subplot(211)
plt.xlabel('t')
plt.ylabel('x(t) or y(t)')
plt.plot(sol.t, sol.y[0], label='x(t)')
plt.plot(sol.t, sol.y[1], label='y(t)')
plt.legend()

plt.subplot(212)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(sol.y[0], sol.y[1])

plt.savefig('solve_ivp_ex7.png')