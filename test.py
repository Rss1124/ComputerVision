import numpy as np

arr = np.array([[1, 8, 3], [2, 8, 6]])
print(arr)
id = np.flatnonzero(arr == 8)
print(id)
