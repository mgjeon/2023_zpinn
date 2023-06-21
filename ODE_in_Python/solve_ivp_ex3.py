import numpy as np 

def dSdt(t, y):
    return np.array([np.exp(-t)])

t_span = (0, 1)

h = 0.1
t = np.arange(t_span[0], t_span[1]+h, h)

S0 = np.array([-1])
# ---------------------------------------
N = len(t) - 1
S = np.zeros(len(t))
S[0] = S0 

for i in range(0, N):
    S[i+1] = S[i] + h*dSdt(t[i], S[i])

# ---------------------------------------
from scipy.integrate import solve_ivp 
sol = solve_ivp(dSdt, t_span, S0, max_step=h)
print(sol)

# ---------------------------------------
from matplotlib import pyplot as plt
fig, ax = plt.subplots()

def analytic_y(t):
    return -np.exp(-t)
y_exact = analytic_y(t)
ax.plot(t, y_exact, 'k-', label='analytic')

ax.plot(t, S, 'bo--', label='Explicit Euler')
ax.set_xlabel('t')
ax.set_ylabel('y(t)')

ax.plot(sol.t, sol.y[0], 'ro--', label='solve_ivp')
ax.set_xlabel('t')
ax.set_ylabel('y(t)')

ax.legend()
plt.tight_layout()
fig.savefig('solve_ivp_ex3.png')