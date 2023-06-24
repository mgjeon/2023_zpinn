import numpy as np 
from scipy.integrate import solve_ivp

def F(t, y):
    return np.array([-y])

t_span = [0, 1]
h = 0.1
t = np.arange(t_span[0], t_span[1]+h, h)

S0 = np.array([1])

sol = solve_ivp(F, t_span, S0, t_eval=t)
print(sol)

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(211)
plt.plot(sol.t, sol.y[0])
plt.xlabel('t')
plt.ylabel('S(t)')

f_exact = lambda t: np.exp(-t)
y_exact = f_exact(t)
plt.subplot(212)
plt.plot(t, sol.y[0]-y_exact)
plt.xlabel('t')
plt.ylabel('S(t) - exp(-t)')

plt.tight_layout()
plt.savefig('solve_ivp_ex6.png')