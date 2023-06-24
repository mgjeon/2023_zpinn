import numpy as np
from scipy.integrate import solve_ivp

def F(t, y):
    return np.array([np.cos(t)])

t_span = [0, np.pi]
N = 1000
t = np.linspace(t_span[0], t_span[1], N)

S0 = np.array([0])

sol = solve_ivp(F, t_span, S0, t_eval=t, rtol = 1e-8, atol = 1e-8)
print(sol)

import matplotlib.pyplot as plt

plt.figure(figsize = (6, 8))

plt.subplot(211)
plt.plot(sol.t, sol.y[0])
plt.xlabel('t')
plt.ylabel('S(t)')

plt.subplot(212)
plt.plot(sol.t, sol.y[0] - np.sin(sol.t))
plt.xlabel('t')
plt.ylabel('S(t) - sin(t)')

plt.tight_layout()
plt.savefig('solve_ivp_ex5.png')