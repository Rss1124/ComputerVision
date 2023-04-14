"""
教程12: 给矩阵中的数据排序
"""

import numpy as np

dt = np.dtype([("name", "S10"), ("age", "i4")])
arr = np.array([("A", 10), ("B", 5), ("C", 20)], dtype=dt)
arr1 = np.sort(a=arr, order="age")
print(arr1)
