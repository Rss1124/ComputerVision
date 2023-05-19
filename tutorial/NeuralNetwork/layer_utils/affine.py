import numpy as np

""" ---------------------------------------- Affine层(全连接层) ---------------------------------------- """
# 笔记:
# Affine层接收输入数据，并将其与权重矩阵相乘，然后添加偏置向量，这个操作可以用数学公式表示为: out = x.dot(w) + b
# Affine层实现了线性变换，将输入数据映射到一个新的空间。
def affine_forward(x, w, b):
    """
    函数功能: affine的前向传播,计算前向传播的输出值,保存缓存

    input:
        - x: numpy数组(N, d_1, ..., d_k)
        - w: 权重矩阵(D, M)
        - b: 偏执项(M,)
    return:
        - out: 前向传播的结果(N, M)
        - cache: 缓存
    """

    out = None
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1)  # 将x转变为(N, D)方便进行矩阵运算
    out = x_reshaped.dot(w) + b
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    """
    函数功能: affine的反向传播,计算当前神经元的梯度

    input:
        - dout: 上一层神经元的梯度(N, M)
        - cache: 上一层神经元的缓存
    return:
        - dx: x的梯度 (N, D)
        - dw: w的梯度 (D, M)
        - db: b的梯度 (M,)
    """

    dx, dw, db = None, None, None
    x, w, b = cache
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1)  # 将x转变为(N, D)方便进行矩阵运算
    dx = dout.dot(w.T).reshape(x.shape)  # 最后需要将dx的shape还原为(N, d_1, ..., d_k)
    dw = x_reshaped.T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db