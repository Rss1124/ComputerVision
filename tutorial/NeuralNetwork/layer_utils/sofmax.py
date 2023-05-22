import numpy as np

""" ---------------------------------------------- softmax损失函数 ---------------------------------------------- """
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