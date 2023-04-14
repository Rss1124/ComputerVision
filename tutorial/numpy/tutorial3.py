"""
教程3: Ndarray数组对象的属性
"""
import numpy as np

arr = np.arange(24)
print("初始化一个一维数组:")
print(arr)
print("将矩阵变为三维:")
arr = arr.reshape(3, 1, 8)
print(arr)
print("将矩阵变为四维:")
arr.shape = (2, 2, 3, 2)
print(arr)
print("矩阵的维度:")
print(arr.ndim)
# 笔记1:
# reshape()和arr.shape都可以改变矩阵的形状,只不过reshape是"函数",shape是"属性",注意两者的写法.
# ndim也是一个"属性",用来描述矩阵的维度.

print("矩阵的元素个数:")
print(arr.size)
print("矩阵元素的数据类型:")
print(arr.dtype)
print("每个元素所占内存的大小:")
print(arr.itemsize)

print("打印数据在内存中的信息:")
print(arr.flags)


