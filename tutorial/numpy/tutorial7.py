"""
教程7: Ndarray数组对象的反转
"""

import numpy as np

arr = np.arange(24).reshape(2, 3, 4)
print("反转之前的矩阵:")
print(arr)
print("反转之前的shape:")
print(arr.shape)
print("\n")
b = np.rollaxis(a=arr, axis=2, start=0)
print("反转之后的矩阵:")
print(b)
print("反转之后的shape")
print(b.shape)
# 笔记1:
# rollaxis函数可以将axis滚动到start位置上去

# 笔记2:
# 反转函数还有T, transpose, 但是rollaxis更加灵活.
# 除此之外还有个swapaxes函数用来交换两个轴的位置
