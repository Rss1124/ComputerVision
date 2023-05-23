import numpy as np

from tutorial.NeuralNetwork.layer_utils.affine import affine_forward, affine_backward
from tutorial.NeuralNetwork.layer_utils.affine_relu import affine_relu_forward, affine_relu_backward
from tutorial.NeuralNetwork.layer_utils.sofmax import softmax_loss

""" --------------------------------- affine_relu + affine + softmax 的两层神经网络 --------------------------------- """
# 两层神经网络:
# 隐藏层是affine_relu 和 affine,损失函数是softmax
class Layer(object):

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        函数功能: 网络层的初始化

        input:
            - input_dim: 输入数据的特征维度
            - hidden_dim: 隐藏层的特征维度
            - num_classes: 特征分类的数量
            - weight_scale: 标准差
            - reg: 正则化系数
        return:
            - 初始化之后的两层网络
        """
        self.params = {}
        self.reg = reg
        """ 初始化符合高斯分布的权重矩阵 """
        W1 = np.random.randn(input_dim, hidden_dim) * weight_scale
        W2 = np.random.randn(hidden_dim, num_classes) * weight_scale
        """ 计算偏差 """
        b1 = np.zeros(hidden_dim)
        b2 = np.zeros(num_classes)
        """ 将各层的权重矩阵和偏差值存储字典中 """
        self.params['W1'] = W1
        self.params['W2'] = W2
        self.params['b1'] = b1
        self.params['b2'] = b2

    def loss(self, X, y=None):
        """
        函数功能: 计算整个网络的损失值,以及每层的梯度

        input:
            - X: 输入数据(N,d1, d2, ..., dn)
            - y: 隐藏层的特征维度
        return:
            - loss: 损失值
            - grads: 各个网络层的梯度(字典格式存储)
        """

        scores = None
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']
        scores_first, cache_first = affine_relu_forward(x=X, w=W1, b=b1)
        scores, cache_second = affine_forward(x=scores_first, w=W2, b=b2)
        if y is None:
            return scores
        else:
            loss, grads = 0, {}
            loss, dout = softmax_loss(x=scores, y=y)
            """ 计算损失 """
            reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
            # 笔记1:
            # 引入因子0.5是为了抵消在梯度计算中微分平方项时产生的2
            loss += reg_loss
            """ 计算并保留每个网络层的梯度 """
            dx_second, dW_second, db_second = affine_backward(dout=dout, cache=cache_second)
            dx_first, dW_first, db_first = affine_relu_backward(dout=dx_second, cache=cache_first)
            """ 将每层的梯度值存放在字典中,注意用于映射的关键字 """
            grads['W1'] = dW_first + self.reg * W1
            grads['W2'] = dW_second + self.reg * W2
            grads['b1'] = db_first
            grads['b2'] = db_second

        return loss, grads



