"""
教程11: 矩阵的运算:加减乘除
"""

import numpy as np

arr1 = np.array([[1, 2], [3, 4]])
print("arr1:")
print(arr1)
arr2 = np.array([[1, 1], [1, 1]])
print("arr2:")
print(arr2)
arr3 = np.add(arr1, arr2)
print("arr1 + arr2:")
print(arr3)
# 笔记1:
# 只有秩为1的矩阵才可以在运算过程中广播

arr4 = np.subtract(arr1, arr2)
print("arr1 - arr2:")
print(arr4)

# arr5 = np.multiply(arr1, arr2)
# print("arr1 x arr2:")
# print(arr5)
#
# arr6 = np.divide(arr2, arr1)
# print("arr2 / arr1:")
# print(arr6)
# 笔记2:
# 函数multiply()和divide()只是矩阵内"对应位置元素"的乘法和除法,不是真正的矩阵乘法

arr7 = np.array([[1, 2, 1], [3, 1, 1]])
print("arr7:")
print(arr7)
arr8 = np.array([[1, 1], [1, 1], [1, 1]])
print("arr8:")
print(arr8)
arr9 = np.dot(arr7, arr8)
print("arr7 x arr8:")
print(arr9)
# 笔记3:
# 函数dot()才是真正的矩阵乘法
