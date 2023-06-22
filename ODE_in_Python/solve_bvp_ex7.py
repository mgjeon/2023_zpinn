import numpy as np 
from scipy.integrate import solve_bvp

def F(x, y, p):
    k = p[0]
    return np.array([y[1], -k*y[0]])

def bc(ya, yb, p):
    k = p[0]
    return np.array([ya[0], yb[0], ya[1]-np.sqrt(k)])

x_span = [0, 2*np.pi]
N = 100
x = np.linspace(x_span[0], x_span[1], N)

y0 = np.ones((2, x.size))
y0[0,1] = 1
y0[0,-2] = -1

@np.vectorize
def find_eigenvalue(p0_init):
    sol = solve_bvp(F, bc, x, y0, p=[p0_init])
    if sol.success:
        return sol
    else:
        return None

eigenvalues = find_eigenvalue(np.arange(0, 100, 0.1))
eigenvalues = np.array([e.p for e in eigenvalues if e is not None])
eigenvalues = np.unique(np.round(eigenvalues, 2)).reshape(-1)
print(eigenvalues)

# import matplotlib.pyplot as plt 

# fig, ax = plt.subplots()

# x_plot = np.linspace(x_span[0], x_span[1], 1000)
# y_plot = sol.sol(x_plot)[0]

# ax.plot(x_plot, y_plot)

# ax.grid(True)
# ax.set_xlabel('x')
# ax.set_ylabel('y(x)')
# ax.axhline(0, color='k', lw=2)
# ax.axvline(0, color='k', lw=2)
# ax.set_xlim((x_span[0]-0.2, x_span[1]+0.2))
# ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi/2))
# plt.savefig('solve_bvp_ex7.png')