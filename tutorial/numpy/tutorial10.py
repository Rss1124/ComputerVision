"""
教程10: 矩阵元素的添加和删除
"""

import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print("打印矩阵arr:")
print(arr)
print("拓展三行元素:")
arr2 = np.append(arr, [[7, 8, 9], [10, 11, 12], [13, 14, 15]], axis=0)
print(arr2)
print("拓展四列元素:")
arr3 = np.append(arr, [[7, 9, 11, 13], [8, 10, 12, 14]], axis=1)
print(arr3)
# 笔记1:
# append函数会返回一个新的矩阵

arr4 = np.insert(arr, 0, [7, 8, 9], axis=0)
print("在第一行插入元素:")
print(arr4)
arr5 = np.insert(arr, 1, [7, 8], axis=1)
print("在第二列插入元素:")
print(arr5)
# 笔记2:
# 如果不加axis参数的话,arr矩阵会被直接压缩成一维数组

arr6 = np.delete(arr5, 0, axis=0)
print("删除矩阵的第一行:")
print(arr6)
arr7 = np.delete(arr5, 1, axis=1)
print("删除矩阵的第二列:")
print(arr7)

