import numpy as np

def softmax_loss_naive(W, X, y, reg):
    """
    函数功能: 使用softmax损失函数计算损失值和梯度矩阵

    :param W: 权重矩阵(M*C的二维矩阵: M表示样本有多少个特征, C表示数据集有多少分类)
    :param X: 训练集(N*M的二维矩阵: N表示有多少样本, M表示样本有多少个特征)
    :param y: 训练集的标签(1*N的二维矩阵)
    :param reg: 正则化系数
    :return: 损失值和梯度矩阵
    """
    num_train = X.shape[0]
    loss = 0
    dW = np.zeros(W.shape)
    for i in range(num_train):
        scores = X[i].dot(W)
        e_scores = np.exp(scores)
        """ 计算loss """
        correct_scores = e_scores[y[i]]  # y[i]: 正确分类在e_scores的索引
        """ softmax损失函数的变形 """
        sub_loss = np.log(np.sum(e_scores)) - np.log(correct_scores)
        loss += sub_loss
        """ 计算梯度 """
        for r in range(W.shape[0]):
            for c in range(W.shape[1]):
                if y[i] != c:
                    dW[r][c] += (e_scores[c] * X[i][r]) / np.sum(e_scores)
                else:
                    dW[r][c] += ((e_scores[c] * X[i][r]) / np.sum(e_scores)) - X[i][r]
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += reg * 2 * W
    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    函数功能: 使用softmax损失函数计算损失值和梯度矩阵(向量化实现)

    :param W: 权重矩阵(M*C的二维矩阵: M表示样本有多少个特征, C表示数据集有多少分类)
    :param X: 训练集(N*M的二维矩阵: N表示有多少样本, M表示样本有多少个特征)
    :param y: 训练集的标签(1*N的二维矩阵)
    :param reg: 正则化系数
    :return: 损失值和梯度矩阵
    """

    num_train = X.shape[0]
    dW = np.zeros(W.shape)
    """ 一次性计算所有分数矩阵 """
    scores = X.dot(W)
    """ 对分数矩阵进行一次指数运算 """
    exp_scores = np.exp(scores)
    """ 获取所有分类的概率 """
    p_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    """ 计算loss """
    loss = -1 * np.log(p_scores[np.arange(scores.shape[0]), y])
    loss = np.sum(loss) / num_train + reg * np.sum(W * W)
    """ 计算梯度 """
    Jacobi = X.T.dot(p_scores)  # 雅可比矩阵
    binary = np.zeros(scores.shape)  # 二元矩阵
    binary[np.arange(scores.shape[0]), y] += 1
    Bias = X.T.dot(binary)  # 偏置矩阵

    dW = (Jacobi - Bias) / num_train + reg * 2 * W

    return loss,dW
