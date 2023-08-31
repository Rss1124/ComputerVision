

""" --------------------------------------- conv + relu + pool 的网络层结构 --------------------------------------- """
from tutorial.NeuralNetwork.layer_utils.conv import conv_forward_im2col, conv_backward_col2im
from tutorial.NeuralNetwork.layer_utils.max_pool import max_pool_forward_im2col, max_pool_backward_col2im
from tutorial.NeuralNetwork.layer_utils.relu import relu_forward, relu_backward


# 笔记1:
# conv_relu_pool是复合的网络层结构，用于构建神经网络模型中的隐藏层。它由三个部分组成：conv层和relu激活函数以及pool层组成

# conv层是该网络的核心部分，它使用一系列的卷积核对输入的数据进行卷积操作，从而提取出图像的特征，每个卷积核会滑动在输入数据上，并在每个
# 图像位置上进行点积操作，最终得到特征输出图

# relu激活函数用来引入非线性变换，从而增加网络的表达能力

# pool层用于减少特征输出图的尺寸，并保留最重要的信息，从而进一步减少计算量。在这里使用的池化操作是最大池化，它在每个池化区域中选择最大
# 的数值作为输出。池化有助于防止过拟合

# 总结：一个完整的卷积神经网络会由多个conv + ReLU + pool序列构成，其中每个序列的卷积核数量逐层递增或递减，用于提取越来越抽象的特征

def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    函数功能: conv_relu_pool网络层的前向传播

    input:
        - x: shape为(N,C,H,W)的图像集
        - w: shape为(F,C,HH,WW)的滤波器集
        - b: shape为(F, )偏执项
        - conv_param: 卷积参数,包含(stride, pad)
        - pool_param: 池化参数,包含(pool_height, pool_width, stride)

    Returns a tuple of:
        - out: 从池化层输出的数据
        - cache: 缓存数据
    """

    conv_out, conv_cache = conv_forward_im2col(x, w, b, conv_param)
    relu_out, relu_cache = relu_forward(conv_out)
    out, pool_cache = max_pool_forward_im2col(relu_out, pool_param)

    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache

def conv_relu_pool_backward(dout, cache):
    """
    函数功能: conv_relu_pool网络层的反向传播

    input:
        - dout: 上一层的梯度
        - cache: 上一层的缓存

    returns a tuple of:
        - dx: x(image)的梯度，shape为(N, C, H, W)
        - dw: w(filter)的梯度，shape为(F, C, HH, WW)
        - db: b的梯度，shape为(F,)
    """

    conv_cache, relu_cache, pool_cache = cache
    pool_dout = max_pool_backward_col2im(dout, cache=pool_cache)
    relu_dout = relu_backward(dout=pool_dout, cache=relu_cache)
    dx, dw, db = conv_backward_col2im(dout=relu_dout, cache=conv_cache)
    return dx, dw, db


""" ------------------------------------------- conv + relu 的网络层结构 ------------------------------------------- """

def conv_relu_forward(x, w, b, conv_param):
    """
    函数功能: conv_relu网络层的前向传播

    input:
        - x: shape为(N,C,H,W)的图像集
        - w: shape为(F,C,HH,WW)的滤波器集
        - b: shape为(F, )偏执项
        - conv_param: 卷积参数,包含(stride, pad)

    Returns a tuple of:
        - out: 从relu激活函数输出的数据
        - cache: 缓存数据
    """

    conv_out, conv_cache = conv_forward_im2col(x, w, b, conv_param)
    out, relu_cache = relu_forward(conv_out)

    cache = (conv_cache, relu_cache)
    return out, cache

def conv_relu_backward(dout, cache):
    """
    函数功能: conv_relu网络层的反向传播

    input:
        - dout: 上一层的梯度
        - cache: 上一层的缓存

    returns a tuple of:
        - dx: x(image)的梯度，shape为(N, C, H, W)
        - dw: w(filter)的梯度，shape为(F, C, HH, WW)
        - db: b的梯度，shape为(F,)
    """

    conv_cache, relu_cache = cache
    relu_dout = relu_backward(dout, cache=relu_cache)
    dx, dw, db = conv_backward_col2im(dout=relu_dout, cache=conv_cache)
    return dx, dw, db