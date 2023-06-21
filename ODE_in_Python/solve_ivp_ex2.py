import numpy as np
from scipy.integrate import solve_ivp 

def f(t, y):
    return np.array([y[1], y[0]+np.sin(t)])

t_span = (0, 5)

y0 = np.array([-1/2, 0])

t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = solve_ivp(f, t_span, y0, t_eval=t_eval)
print(sol)

from matplotlib import pyplot as plt
fig, axes = plt.subplots(nrows=2, ncols=1)

axes[0].plot(sol.t, sol.y[0], label='solve_ivp')
axes[0].set_xlabel('t')
axes[0].set_ylabel('y(t)')

def analytic_y(t):
    return -0.5*(np.exp(-t) + np.sin(t))

y_exact = analytic_y(t_eval)
axes[0].plot(t_eval, y_exact, '--', label='analytic')
axes[0].legend()

axes[1].plot(sol.t, sol.y[1])
axes[1].set_xlabel('t')
axes[1].set_ylabel("y'(t)")

plt.tight_layout()
fig.savefig('solve_ivp_ex2.png')