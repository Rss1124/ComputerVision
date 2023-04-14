"""
教程9: 矩阵连接和分割
"""

import numpy as np

arr = np.arange(4).reshape(2, 2)
print("打印arr:")
print(arr)
arr2 = np.array([[4, 5], [6, 7]])
print("打印arr2:")
print(arr2)
arr3 = np.concatenate((arr, arr2), axis=0)
print("在y轴上连接arr和arr2:")
print(arr3)
print("在x轴上连接arr和arr2:")
arr4 = np.concatenate((arr, arr2), axis=1)
print(arr4)
# 笔记1:
# 在该实例中,axis=0代表y轴,axis=1代表x轴

# 笔记2:
# 还有stack函数,hstack函数用来连接矩阵

arr5 = np.arange(12).reshape(2, 2, 3)
print("打印arr5:")
print(arr5)
print("切割arr5:")
arr6 = np.split(arr5, indices_or_sections=2, axis=1)
print(arr6)
# 笔记3:
# 第二个参数indices_or_sections表示要将矩阵分割为几份
# split函数返回的是一个list
# 笔记4:
# 还有水平切割函数(hsplit),垂直切割函数(vsplit)



