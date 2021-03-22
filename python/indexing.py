import numpy as np

a = np.arange(1, 49).reshape(3, 4, 4)
print(a)

print("\nIndexing:")
index = np.full((4, 4), 1, dtype=np.int32)
index[0, 0] = 0

b = np.choose(index, a)
print(b)