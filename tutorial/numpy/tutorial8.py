"""
教程8: 修改矩阵维度
"""

import numpy as np

arr = np.arange(1, 9).reshape(2, 4)
print("打印矩阵arr:")
print(arr)
print("矩阵升维:")
arr = np.expand_dims(arr, axis=1).reshape(2, 2, 2)
print(arr)
# 笔记1:
# expand_dims会在指定位置插入一个轴,从而达到升维的效果

arr = arr.reshape(2, 4, 1)
print("打印矩阵arr:")
print(arr)
arr = np.squeeze(arr)
print("矩阵降维:")
print(arr)
# 笔记2:
# 矩阵降维需要严格的要求,只有某一维的高度为1的时候才可以
