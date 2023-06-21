import numpy as np
from scipy.integrate import solve_ivp  

# ODE system
def f(t, y):
    return np.array([y[1], -3*y[0] + np.sin(t)])

# Time domain
t_span = (0, 5)

# Initial Condition
y0 = np.array([-np.pi/2, np.pi])

sol = solve_ivp(f, t_span, y0, max_step=0.01)
print(sol)

from matplotlib import pyplot as plt

fig, axes = plt.subplots(nrows=2, ncols=1)

axes[0].plot(sol.t, sol.y[0])
axes[0].set_xlabel('t')
axes[0].set_ylabel('y(t)')

axes[1].plot(sol.t, sol.y[1])
axes[1].set_xlabel('t')
axes[1].set_ylabel("y'(t)")

plt.tight_layout()
fig.savefig('solve_ivp_ex1.png')
