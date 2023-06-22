import numpy as np 
from scipy.integrate import solve_bvp

# ODE system
def f(x, y):
    return np.array([y[1], -9*y[0]+np.sin(x)])

# BC
def bc(ya, yb):
    return np.array([ya[1]-5, yb[0]+(5/3)])

# Space domain
x_nums = 1000
x_eval = np.linspace(0, np.pi, x_nums)

# Initial guess
y0 = np.ones((2, x_nums))

sol = solve_bvp(f, bc, x_eval, y0)
print(sol)

from matplotlib import pyplot as plt

fig, axes = plt.subplots(nrows=2, ncols=1)

axes[0].plot(sol.x, sol.y[0])
axes[0].set_xlabel('t')
axes[0].set_ylabel('y(t)')

axes[1].plot(sol.x, sol.y[1])
axes[1].set_xlabel('t')
axes[1].set_ylabel("y'(t)")

plt.tight_layout()
fig.savefig('solve_bvp_ex1.png')

