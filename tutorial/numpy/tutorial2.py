"""
教程2: 创建初始化Ndarray数组对象
"""

import numpy as np

test_list = [1, 2.5, 3, 4, 5, 6]
arr = np.array(test_list)
print("生成列表:")
print(test_list)
print(type(test_list))
print("生成numpy的一维数组:")
print(arr)
print(type(arr))
# 笔记1:
# numpy数组会要求元素的数据类型一致,如果不一样会依照(str>float>int)的优先级,将数据类型进行转换

arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='f')
print("生成numpy的二维数组:")
print(arr2)
print(type(arr2))
# 笔记2:
# 可以通过(参数:dtype),来指定数组的数据类型

student = np.dtype([("name", 'S20'), ("age", "i8"), ("grades", "i8")])
arr3 = np.array([[("timy", 10, 80)], [("piter", 12, 75)], [("tom", 11, 78)]], dtype=student)
print("生成一个'结构化数据类型'的3x1的矩阵")
print(arr3)

arr4 = np.asarray([(1, 2, 3), (4, 5)], dtype=object)
print("生成一个包含元组数据的一维矩阵:")
print(arr4)
print(type(arr4))
# 笔记3:
# 如果要生成"不对称的高维数组"比如[[1, 2, 3], [4, 5]]会报错。
# 此时的解决方法是使用"dtype"将数据类型转变为"object",这样就会得到一个一维数组

arr5 = np.empty([2, 2], dtype=student)
print("生成一个2x2的矩阵")
print(arr5)
# 笔记4:
# [2, 2]表示矩阵的shape,也可以使用元组(2, 2)
# 除了empty函数还有zeros,ones函数.不同的函数初始化的数据不一样,
# zeros是初始化为0,ones是初始化为1.而empty函数是随机从内存中取值.
# 除此之外还可以指定数据类型,可以是结构体也可以是普通数据类型.
# 补充函数:full(), eye()

s = b"tom is a cat"
arr6 = np.frombuffer(s, dtype='S3')
print("使用buffer流生成一个一维数组:")
print(arr6)
# 笔记5:
# 这里用到了一个新的数据(流),在字符串开头加上一个字符b,即可表示这个数据是一个流.
# 该函数会将字符串分割成一个一维数组,数组的元素个数取决于你是如何分割的.
# 除了dtype参数,还有count, offset,这些参数都会影响最终的效果

arr7 = np.linspace(1, 8, 4)
print("生成一个等差数列:")
print(arr7)

arr8 = np.logspace(1, 10, 10, dtype="f", base=2)
print("生成一个等比数列:")
print(arr8)
# 笔记6:
# 感觉linspace和logspace两个函数不太好用

arry9 = np.random.rand(2, 3, 2)
print("随机生成一个矩阵:")
print(arry9)
# 笔记7:
# 还有random(),randint(),randn(),normal()

# 笔记8:
# "numpy数组"比"python中自带的列表"效率高
