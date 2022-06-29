import numpy as np

A = [[1, 2, 3],
     [2, 3, 4],
     [4, 5, 6],
     [1, 1, 1]]

U, S, VT = np.linalg.svd(A, full_matrices=True)

print(U, "\n\n")
print(S, "\n\n")
print(VT, "\n")