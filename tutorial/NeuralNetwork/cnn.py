
""" ------------------------ conv_relu_pool + affine_relu + affine + softmax 的三层神经网络 ------------------------ """
import numpy as np

from tutorial.NeuralNetwork.layer_utils.affine import affine_forward, affine_backward
from tutorial.NeuralNetwork.layer_utils.affine_relu import affine_relu_forward, affine_relu_backward
from tutorial.NeuralNetwork.layer_utils.conv_relu_pool import conv_relu_pool_forward, conv_relu_pool_backward
from tutorial.NeuralNetwork.layer_utils.sofmax import softmax_loss


# 三层神经网络：
# 隐藏层：conv_relu_pool(size为2*2)、affine_relu、affine
# 损失函数：softmax

class ThreeLayerConvNet(object):

    def __init__(
            self,
            input_dim=(3,32,32),
            num_filters=32,
            filter_size=7,
            hidden_dim=100,
            num_classes=10,
            weight_scale=1e-3,
            reg=0.0,
            dtype=np.float32,
    ):
        """
        函数功能: 神经网络的初始化

        input:
            - input_dim: 输入数据的特征维度,默认为3通道,32*32像素
            - num_filters: 滤波器的数量,默认为32个
            - filter_size: 滤波器的尺寸,默认为7*7像素
            - hidden_dim: 隐藏层的特征维度,默认为100
            - num_classes: 特征分类的数量,默认为10
            - weight_scale: 标准差,默认为1e-3
            - reg: 正则化系数,默认为0
            - dtype: numpy的数据类型,默认为float32
        return:
            - 初始化之后的两层网络
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        C, H, W = input_dim
        conv_OH = 1 + (H + 2 * conv_param["pad"] - filter_size) // conv_param["stride"]
        conv_OW = 1 + (W + 2 * conv_param["pad"] - filter_size) // conv_param["stride"]
        pool_OH = 1 + (conv_OH - pool_param["pool_height"]) // pool_param["stride"]
        pool_OW = 1 + (conv_OW - pool_param["pool_width"]) // pool_param["stride"]

        """ 初始化符合高斯分布的权重矩阵 """
        W1 = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
        W2 = np.random.randn(num_filters * pool_OH * pool_OW, hidden_dim) * weight_scale
        W3 = np.random.randn(hidden_dim, num_classes) * weight_scale

        """ 初始化偏置矩阵 """
        b1 = np.zeros(num_filters)
        b2 = np.zeros(hidden_dim)
        b3 = np.zeros(num_classes)

        """ 将各层的权重矩阵和偏差值存储字典中 """
        self.params['W1'] = W1
        self.params['W2'] = W2
        self.params['W3'] = W3
        self.params['b1'] = b1
        self.params['b2'] = b2
        self.params['b3'] = b3

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        函数功能: 计算整个网络的损失值,以及每层的梯度

        input:
            - X: 输入数据(N,C,H,W)
            - y: 数据集对应的标签数据
        return:
            - loss: 损失值
            - grads: 各个网络层的梯度(字典格式存储)
        """

        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        filter_size = W1.shape[2]

        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None

        """ 计算第一层(conv_relu_pool) """
        scores_first, cache_first = conv_relu_pool_forward(x=X, w=W1, b=b1, conv_param=conv_param, pool_param=pool_param)
        """ 计算第二层(affine_relu) """
        scores_second, cache_second = affine_relu_forward(x=scores_first, w=W2, b=b2)
        """ 计算第三层(affine) """
        scores, cache_third = affine_forward(x=scores_second, w=W3, b=b3)

        if y is None:
            return scores  # 计算整个神经网络的分数
        else:
            loss, grads = 0, {}
            loss, dout = softmax_loss(x=scores, y=y)

            """ 计算整个神经网络的损失 """
            reg_loss = 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
            loss += reg_loss

            """ 计算各层的梯度 """
            dx_third, dw_third, db_third = affine_backward(dout=dout, cache=cache_third)
            dx_second, dw_second, db_second = affine_relu_backward(dout=dx_third, cache=cache_second)
            dx_first, dw_first, db_first = conv_relu_pool_backward(dout=dx_second, cache=cache_first)

            """ 将各层的梯度变化(dx)存储下来,注意用于映射的关键字 """
            grads['W1'] = dw_first + self.reg * W1
            grads['W2'] = dw_second + self.reg * W2
            grads['W3'] = dw_third + self.reg * W3
            grads['b1'] = db_first
            grads['b2'] = db_second
            grads['b3'] = db_third

        return loss, grads



