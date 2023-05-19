import numpy as np

""" ------------------------------------------ affine + relu 的网络层结构 ------------------------------------------ """
# 笔记1:
# affine_relu是一种常用的网络层结构,用于构建神经网络模型中的隐藏层.它由两个部分组成: Affine层和ReLU激活函数.
# 将Affine层和ReLU激活函数结合在一起形成affine_relu层,可以在神经网络的隐藏层中引入非线性变换.这对于神经网络的表达能力非常重要,
# 因为它允许网络学习更加复杂的特征和决策边界.ReLU激活函数还有助于缓解梯度消失问题,并提供更好的梯度传播,从而加快训练过程.
#
# 总结而言,affine_relu层的设计旨在引入非线性变换和激活函数,以增强神经网络的表达能力,并改善梯度传播的效果.
# 这有助于提高神经网络的性能和学习能力.

def affine_relu_forward(x, w, b):
    """
    函数功能: a + r 神经网络的前向传播,计算前向传播的输出值,保存缓存

    input:
        - x: 训练数据,会输入进affine
        - w: 权重矩阵
        - b: 偏置项
    return:
        - out: 从relu输出的数据
        - cache: 缓存数据
    """

    affine_out, affine_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(affine_out)
    cache = (affine_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """
    函数功能: a + r 神经网络的反向传播

    input:
        - dout: 上一层的梯度
        - cache: 上一层的缓存
    return:
        - dx: affine_x的梯度
        - dw: affine_w的梯度
        - db: affine_b的梯度
    """

    affine_cache, relu_cache = cache
    relu_dout = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(relu_dout, affine_cache)
    return dx, dw, db
