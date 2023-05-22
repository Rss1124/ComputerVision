import numpy as np

""" ---------------------------------------------- svm_loss损失函数 ---------------------------------------------- """
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
