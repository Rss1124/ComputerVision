import numpy as np
from numpy import sqrt


def sgd(w, dw, config=None):
    """
    函数功能: 更新权重矩阵

    input:
        - w: 权重矩阵
        - dw: 梯度矩阵
        - config: 包含超参数值(如学习率、动量等)的字典。如果更新规则需要在多次迭代中缓存值，那么config也将保存这些缓存值。
    return:
        - w: 新的权重矩阵
        - config: 包含学习率的字典
    """

    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    w -= config["learning_rate"] * dw

    return w, config

def sgd_momentum(w, dw, config=None):
    """
    函数功能: 使用带动量的随机梯度下降算法来更新权重矩阵,动量会累积之前的梯度信息,从而减少更新的方向变化,使优化过程更平稳

    input:
        - w: 权重矩阵
        - dw: 梯度矩阵
        - config: 包含超参数值(如学习率、动量等)的字典。如果更新规则需要在多次迭代中缓存值，那么config也将保存这些缓存值。
    return:
        - next_w: 新的权重矩阵
        - config: 包含学习率的字典
    """

    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)  # 学习率
    config.setdefault("momentum", 0.9)  # 动量系数
    v = config.get("velocity", np.zeros_like(w))
    # 笔记1:
    # momentum是动量系数，v是动量项（初始化为零矩阵）

    next_w = None

    """ 更新动量项 """
    v = config['momentum'] * v - config['learning_rate'] * dw
    # 笔记2:
    # 动量更新规则：v = momentum * v - learning_rate * dw

    config["velocity"] = v

    """ 更新权重矩阵 """
    next_w = w + v

    return next_w, config

def rmsprop(w, dw, config=None):
    """
    函数功能: 一种基于梯度的优化算法，用于调整神经网络参数的学习率

    input:
        - w: 权重矩阵
        - dw: 梯度矩阵
        - config: 包含超参数值(如学习率、动量等)的字典。如果更新规则需要在多次迭代中缓存值，那么config也将保存这些缓存值。
    return:
        - next_w: 新的权重矩阵
        - config: 包含学习率的字典
    """

    if config is None:
        config = {}

    """ 初始化学习率 """
    config.setdefault("learning_rate", 1e-2)

    """ 初始化衰减率(RMSProp参数) """
    config.setdefault("decay_rate", 0.99)

    """ 初始化累积梯度平方项 """
    config.setdefault("cache", np.zeros_like(w))

    config.setdefault("epsilon", 1e-8)

    next_w = None

    learning_rate = config["learning_rate"]
    decay_rate = config["decay_rate"]
    epsilon = config["epsilon"]
    cache = config["cache"]

    """ 更新累计梯度平方项 """
    cache = decay_rate * cache + (1 - decay_rate) * (dw**2)

    next_w = w - learning_rate / (np.sqrt(cache) + epsilon) * dw
    # 笔记1:
    # RMSProp的主要思想是根据之前梯度的大小来调整每个参数的学习率，从而适应不同参数的更新速度。
    # RMSProp的关键点在于，它根据历史梯度的大小来调整每个参数的学习率。
    # 如果某个参数的梯度变化较大，那么对应的学习率会相应减小，从而避免了梯度爆炸的问题。
    # 同时，它也能够自适应地调整参数的学习率，适用于不同参数的更新速度差异较大的情况。

    config["cache"] = cache

    return next_w, config


def adam(w, dw, config=None):
    """
    函数功能: adam一种基于梯度的优化算法,是一种基于梯度的优化算法,结合了动量法和RMSProp算法

    input:
        - w: 权重矩阵
        - dw: 梯度矩阵
        - config: 包含超参数值(如学习率、动量等)的字典。如果更新规则需要在多次迭代中缓存值，那么config也将保存这些缓存值。
    return:
        - next_w: 新的权重矩阵
        - config: 包含学习率的字典
    """

    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None

    learning_rate = config["learning_rate"]  # 学习率
    beta1 = config["beta1"]  # 动量参数β1
    beta2 = config["beta2"]  # RMSProp参数β2
    epsilon = config["epsilon"]
    m = config["m"]  # 动量项（初始化为零矩阵）
    v = config["v"]  # RMSProp项（初始化为零矩阵）
    t = config["t"] + 1  # 迭代次数

    """ 更新动量项 """
    m = beta1 * m + (1-beta1) * dw

    """ 更新RMSProp项 """
    v = beta2 * v + (1-beta2) * (dw**2)
    # 笔记1:
    # 它使用了动量的概念来考虑之前的梯度变化趋势,并使用RMSProp来调整每个参数的学习率

    m_hat = m/(1 - beta1**t)
    v_hat = v/(1 - beta2**t)
    # 笔记2:
    # 在优化算法中，动量项和RMSProp项都是通过计算移动平均来更新权重。移动平均是对过去一段时间内的梯度或梯度平方进行平均，
    # 以便更好地估计梯度的整体趋势。然而，在初始迭代阶段，由于计算的时间窗口较小，移动平均可能会受到初始值或初始数据的影响，
    # 导致计算出的移动平均值具有一定的偏差。
    #
    # 例如，初始迭代时的梯度可能波动较大，如果直接用这些波动的梯度计算移动平均，可能会使移动平均值偏离真实的梯度变化趋势。
    # 这种偏差可能会影响算法的稳定性和收敛速度。
    #
    # 为了减轻这种初始偏差的影响，优化算法引入了偏差修正项。通过在计算移动平均时除以修正因子，可以使移动平均在初始迭代阶段的影响减小，
    # 更准确地反映梯度的真实变化趋势。这样，算法在初始阶段就能更稳定地进行权重更新，从而加速收敛并提高性能

    # 笔记3:
    # 移动平均是一种统计方法，用于平滑时间序列数据或序列数据中的噪声，以便更好地观察数据的趋势和整体变化。
    # 在优化算法中，移动平均常用于估计梯度的整体趋势，从而更稳定地进行权重更新。
    #
    # 在时间序列数据中，移动平均是指在一段时间内取数据的平均值，然后将时间窗口滑动一步，继续计算下一个时间段的平均值，以此类推。
    # 这种方法可以减少噪声的影响，使数据的变化趋势更为明显。

    """ 计算新的权重矩阵 """
    next_w = w - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    """ 更新动量项,RMSProp项以及迭代次数t"""
    config["m"] = m
    config["v"] = v
    config["t"] = t

    return next_w, config
