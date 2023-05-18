import numpy as np

def affine_forward(x, w, b):
    """
    函数功能: 矩阵乘法的前向传播,计算分数矩阵,获取缓存数据

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
    函数功能: 矩阵乘法的反向传播,计算当前神经元的梯度

    input:
        - dout: 上一层神经元的梯度(N, M)
        - cache: 上一层神经元的缓存
    return:
        - dx: x的梯度 * dout (N, D)
        - dw: w的梯度 * dout (D, M)
        - db: b的梯度 * dout (M,)
    """

    dx, dw, db = None, None, None
    x, w, b = cache
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1)  # 将x转变为(N, D)方便进行矩阵运算
    dx = dout.dot(w.T).reshape(x.shape)  # 最后需要将dx的shape还原为(N, d_1, ..., d_k)
    dw = x_reshaped.T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db

def relu_forward(x):
    """
    函数功能: relu的前向传播,计算输出值,获取缓存

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

def svm_loss(x, y):
    """
    函数功能: 基于svm_loss来计算当前神经元的损失和分数矩阵的梯度

    input:
        - x: 基于svm_loss的得分矩阵(N,C)
        - y: 标签(N,)
    return:
        - loss: x的损失
        - dx: 分数矩阵的梯度(N,C)
    """

    loss = None
    dx = None
    N = x.shape[0]
    scores = x.copy()
    correct_scores = scores[np.arange(N), y].reshape(N, 1)
    margins = np.maximum(0, scores - correct_scores + 1)
    margins[np.arange(N), y] = 0
    """ 计算loss """
    loss = np.sum(margins) / N
    """ 计算分数矩阵的梯度 """
    dx = margins
    dx[margins > 0] = 1
    row_sum = np.sum(dx, axis=1)
    dx[np.arange(N), y] -= row_sum
    dx /= N
    return loss, dx

def softmax_loss(x, y):
    """
    函数功能: 基于softmax_loss来计算当前神经元的损失和分数矩阵的梯度

    input:
        - x: 基于softmax_loss的得分矩阵(N,C)
        - y: 标签(N,)
    return:
        - loss: x的损失
        - dx: 分数矩阵的梯度(N,C)
    """

    loss = None
    dx = None
    N = x.shape[0]
    scores = x.copy()
    exp_scores = np.exp(scores)
    correct_exp_scores = exp_scores[np.arange(N), y]
    p_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    """ 计算loss """
    loss = -1 * np.log(p_scores[np.arange(N), y])
    loss = np.sum(loss) / N
    """ 计算分数矩阵的梯度 """
    dx = np.zeros_like(scores)
    dx += np.sum(exp_scores, axis=1).reshape(N,1)
    dx = 1 / dx
    correct_exp_scores = 1 / correct_exp_scores
    dx[np.arange(N), y] -= correct_exp_scores
    dx = dx * exp_scores
    dx /= N
    return loss, dx
