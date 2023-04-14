"""
教程6: Ndarray数组对象的遍历
"""

import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print("打印arr矩阵:")
print(arr)
print("遍历arr矩阵:")
for x in np.nditer(arr):
    print(x, end=",")
print("\n")
arr1 = arr.T
print("打印arr的转置矩阵arr1:")
print(arr1)
print("遍历arr1矩阵:")
for x in np.nditer(arr1):
    print(x, end=",")
print("\n")
# 笔记1:
# 由于arr在内存中默认排序为"行优先",且转置操作不会影响内存中的排序,所以arr和arr1的遍历结果相同

arr2 = arr.copy(order="F")
print("改变arr矩阵在内存中的存储方式:")
print(arr2)
print("打印arr:")
print(arr)
print("遍历矩阵:")
for x in np.nditer(arr2, op_flags=["readwrite"]):
    x = x * 2
    print(x, end=",")
print("\n")
# 笔记2:
# order参数用来表示"内存"存储矩阵的方式,F表示列优先,C表示行优先
# 即使改变了内存中的存储顺序,但是矩阵的shape却不会变

# 笔记3:
# 在迭代器函数nditer()中添加参数op_falgs,可以更改矩阵中的数据


