"""
教程4: Ndarray数组对象的切割
"""

import numpy as np

arr = np.arange(10)
print("初始化一个一维矩阵:")
print(arr)
print("通过下标2来定位矩阵元素:")
print(arr[2])
# s = slice(2, 7, 2)
# print(type(s))
# print(arr[s])
arr = arr[2:7:2]
print("对一维矩阵进行切片:")
print(arr)
# 笔记1: 还可以使用slice对象对一维数组进行切片,请看上述注释代码

arr = np.arange(9)
arr = arr.reshape(3, 3)
print("初始化一个二维矩阵:")
print(arr)
print("输出arr[2]:")
print(arr[2])
print("输出arr[2][2]:")
print(arr[2][2])
# 笔记2: 如果对一个二维矩阵用一个下标去定位,会得到某一行的所有数据

arr2 = np.append(arr[:2], arr[2:], axis=0)
print("使用冒号来对二维矩阵进行切割:")
print(arr2)
print("切割后的矩阵的shape:")
print(arr2.shape)

arr3 = arr[..., 1]
print("初始矩阵:")
print(arr)
print("打印第二列的元素")
print(arr3)
print("打印第三行的元素")
arr4 = arr[2, ...]
print(arr4)

print("初始矩阵:")
print(arr)
rows = np.array([[0, 0], [2, 2]])
cols = np.array([[0, 2], [0, 2]])
print("使用高级检索:")
print(arr[rows, cols])
# 笔记3：高级检索还有一个布尔操作符arr[arr>1]
