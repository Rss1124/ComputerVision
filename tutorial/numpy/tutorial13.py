"""
教程13: 查找矩阵中的数据
"""

import numpy as np

arr = np.array([[1, 2, 3], [6, 5, 4], [8, 7, 9]])
print("arr:")
print(arr)
print("选出矩阵中大于5且小于7的元素:")
arr1 = np.where((arr < 7) & (arr > 5))
print(arr1)
