import numpy as np

w = 4 

h = 0.1
t_span = (0, 5)
t = np.arange(t_span[0], t_span[1]+h, h)

def dSdt(t, y):
    return np.array([y[1], -(w**2)*y[0]])

S0 = np.array([1, 0])

# ---------------------------------------
from numpy.linalg import inv
N = len(t)

SE = np.zeros((2, N))
SI = np.zeros((2, N))
ST = np.zeros((2, N))

SE[:, 0] = S0 
SI[:, 0] = S0 
ST[:, 0] = S0 

ME = np.array([[1, h], [-w**2*h, 1]])
MI = inv(np.array([[1, -h], [w**2*h, 1]]))
MT = np.dot(inv(np.array([[1, -h/2], [(w**2*h)/2, 1]])), np.array([[1, h/2], [-(w**2*h)/2, 1]]))

for i in range(0, N-1):
    SE[:, i+1] = np.dot(ME, SE[:, i])
    SI[:, i+1] = np.dot(MI, SI[:, i])
    ST[:, i+1] = np.dot(MT, ST[:, i])

# ---------------------------------------
from scipy.integrate import solve_ivp
sol = solve_ivp(dSdt, t_span, S0, t_eval=t)
print(sol)

# ---------------------------------------
from matplotlib import pyplot as plt
fig, ax = plt.subplots()

def analytic_y(t):
    return np.cos(w*t)

t_eval = np.linspace(t_span[0], t_span[1], 1000)
y_exact = analytic_y(t_eval)
ax.plot(t_eval, y_exact, 'k-', label='analytic')

ax.plot(sol.t, sol.y[0], '--', label='solve_ivp')
ax.set_xlabel('t')
ax.set_ylabel('y(t)')

ax.plot(t, SE[0, :], 'b-', label='Explicit Euler')
ax.plot(t, SI[0, :], 'g:', label='Implicit Euler')
ax.plot(t, ST[0, :], 'r--', label='Trapezoidal')

ax.set_ylim([-3, 3])

ax.legend()
plt.tight_layout()
fig.savefig('solve_ivp_ex4.png')
