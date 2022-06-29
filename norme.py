import numpy as np
import math

A = np.tile(0.0, (1402,1402))
delta = 0.001
xi = yi = lambda i: -0.7 + (delta * (i - 1))


for i in range (1, 1402):
  for j in range (1, 1402):
    A[i, j] = math.sqrt(1 - xi(i)**2 - yi(j)**2)

U, S, VT = np.linalg.svd(A, full_matrices = True)

A2 = np.zeros((len(U), len(VT)))
A2 += S[0] * np.outer(U.T[0], VT[0])
A2 += S[1] * np.outer(U.T[1], VT[1])

print(np.linalg.norm(A-A2), '\n')