import numpy as np

A = np.ones((4, 4, 3))
w = np.array([1, 2, 3]);

B = np.dot(A, w)
print(B)