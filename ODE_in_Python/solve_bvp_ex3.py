import numpy as np

N = 10
h = (5-0)/N

A = np.eye(N+1, k=1) + np.eye(N+1, k=-1) - 2*np.eye(N+1, k=0)
A[0, :] = np.concatenate(([1], np.zeros(N)))
A[-1, :] = np.concatenate((np.zeros(N), [1]))
print(A)

b = np.zeros(N+1)
b[1:-1] = -9.8*h**2
b[-1] = 50
print(b)

y = np.linalg.solve(A, b)
t = np.arange(0, 5+h, h)

y_n1 = -9.8*h**2 + 2*y[0] - y[1]
y_p0 = (y[1]-y_n1)/(2*h)
print(f"y'(0) = {y_p0}")


import matplotlib.pyplot as plt 
plt.figure()
plt.plot(t, y)
plt.xlabel('t')
plt.ylabel('y(t)')

plt.plot(5, 50, 'ro')
plt.plot(0, 0, 'ro')
plt.savefig('solve_bvp_ex3.png')