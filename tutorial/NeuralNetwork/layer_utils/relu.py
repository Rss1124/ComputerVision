import numpy as np

""" ---------------------------------------------- ReLU激活函数 ---------------------------------------------- """
# 笔记:
# ReLU（Rectified Linear Unit）激活函数是一种非线性函数，被广泛用于神经网络中。ReLU函数的定义是f(x) = max(0, x)，
# 即将小于零的输入值变为零，保持非负输入值不变。ReLU函数引入了非线性性质，使得神经网络可以学习更加复杂的函数映射。
def relu_forward(x):
    """
    函数功能: relu的前向传播,计算前向传播的输出值,保存缓存

    input:
        - x: numpy数组(N, d_1, ..., d_k)
    return:
        - out: 前向传播的结果(N, d_1, ..., d_k)
        - cache: 缓存
    """

    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    """
    函数功能: relu的反向传播,计算当前神经元的梯度

    input:
        - dout: 上一层神经元的梯度(N, M)
        - cache: 上一层神经元的缓存
    return:
        - dx: x的梯度(N, d_1, ..., d_k)
    """

    dx = None
    x = cache
    dx = dout * (x > 0)
    # 笔记1:
    # (x > 0)返回的是一个boolean数值
    # 如果x>0则返回1, 否则返回0
    return dx