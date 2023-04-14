"""
教程1: numpy数据类型
"""

import numpy as np

dt = np.dtype(np.int64)
print(dt)
print(type(dt))

dt = np.dtype("i8")
print(dt)
print(type(dt))
# 笔记1:
# i1代表int8, i2代表int16, i4代表int32, i8代表int64

student = np.dtype([("name", "S20"), ("age", "i4"), ("grades", "i4")])
print(student)
print(type(student))
# 笔记2:
# (numpy中的数据类型以及对应符号):[(b--布尔型),(i--有符号位的整型),(u--无符号位的整型),(f--浮点型),(c--复数浮点型)
# (m--时间间隔),(M--日期时间),(O--对象),(S,a--字符串),(U--Unicode),(V--原始数据)
