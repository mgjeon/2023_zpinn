import numpy as np 
from scipy.integrate import solve_bvp

def find_P(n, m):

    # Define BVP (Low and Lou 1990)
    # a2 -> eigenvalue
    # S = [P, dP/dmu]
    # F = dSdmu
    # 
    # dP/dmu = 10 at mu = -1
    def F(mu, S, p):
        a2 = p[0]
        P = S[0]
        dP_dmu = S[1]
        d2P_dmu2 = (-1)*( n*(n+1)*P + a2*((1+n)/n)*P**(1+2/n) )/(1-mu**2 + 1e-6)
        return [dP_dmu, d2P_dmu2]

    def bc(S_left, S_right, p):
        P_left = S_left[0]
        dP_left = S_left[1]
        P_right = S_right[0]
        return [P_left, P_right, dP_left-10]

    mu_span = [-1, 1]
    N = 100
    mu = np.linspace(mu_span[0], mu_span[1], N)

    # For given m, use different initial guess
    if m % 2 == 0:
        P_init = np.cos(mu * (m + 1) * np.pi / 2)
    else:
        P_init = np.sin(mu * (m + 1) * np.pi / 2)

    # For initial guess of dP/dmu, just use BC value
    dP_init = 10*np.ones_like(mu)
    S_init = np.vstack([P_init, dP_init])
    
    # For each initial eigenvalue, solve the problem.
    # If it is successful, return that otherwise do not return.
    # np.vectorize -> for loop & return type : array
    @np.vectorize
    def solve_eigenvalue_problem(a2_init):
        sol = solve_bvp(F, bc, mu, S_init, p=[a2_init], tol=1e-6)
        if sol.success == True:
            return sol
        else:
            return None 
        
    a2_init_list = np.linspace(0, 10, 100, dtype=np.float32)
        
    results = solve_eigenvalue_problem(a2_init_list)
    eigenvalues = np.array([sol.p for sol in results if sol is not None])

    # round & unique value & sorting
    eigenvalues = np.sort(np.unique(np.round(eigenvalues, 4)))
    
    # The smallest value for given m is desired eigenvalue
    eigenvalue = eigenvalues[0]
    # If this eigenvalue is zero for nonzero m, choose the next big eigenvalue
    if m > 0:
        if not (eigenvalue > 0):
            eigenvalue = eigenvalues[1]

    # Solve again with that eigenvalue
    sol = solve_eigenvalue_problem([eigenvalue])[0]
    
    return sol.sol, sol.p[0]


n = 1

import matplotlib.pyplot as plt 
fig, ax = plt.subplots(figsize=(8, 6))
mu_plot = np.linspace(-1, 1, 1000)
ax.grid(True)
ax.axhline(0, color='k', lw=2)
ax.axvline(0, color='k', lw=2)
ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'P($\mu$)')

for m in [0, 1, 2]:
    S, a2 = find_P(n, m)
    P_plot = S(mu_plot)[0]
    if a2 < 1e-3:
        P_label = 'P' r'$_{' f'{n}, {m}' r'}(\mu)$ with $a^2' r'_{' f'{n}, {m}' r'}$ = 0'
    else:
        P_label = 'P' r'$_{' f'{n}, {m}' r'}(\mu)$ with $a^2' r'_{' f'{n}, {m}' r'}$ = ' f'{a2:.3g}'
    ax.plot(mu_plot, P_plot, label=P_label)

fig.legend()
fig.savefig('low_out_ode.png', dpi=300)
