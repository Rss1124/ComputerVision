import numpy as np


def svm_loss_naive(W, X, y, reg):
    """
    函数功能: 计算损失和梯度矩阵(一层for循环)

    :param W: 权重矩阵(M*C的二维矩阵: M表示样本有多少个特征, C表示数据集有多少分类)
    :param X: 训练集(N*M的二维矩阵: N表示有多少样本, M表示样本有多少个特征)
    :param y: 训练集的标签(1*N的二维矩阵)
    :param reg: 正则化系数
    :return: 损失值和梯度矩阵
    """
    dW = np.zeros(W.shape)
    num_classes = W.shape[1]  # 分类的数量C
    num_train = X.shape[0]  # 训练集的数据量N
    loss = 0.0
    for i in range(num_train):
        # 计算每个训练数据的不同分类的分数
        scores = X[i].dot(W)
        # 找到训练数据的正确分类的分数
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            else:
                margin = scores[j] - correct_class_score + 1
            if margin > 0:
                """ 计算损失 """
                loss += margin
                """ 计算梯度 """
                dW[:, j] += X[i].T
                dW[:, y[i]] -= X[i].T
                # 笔记1:
                # loss = W0*(X0+X1+...+Xn) + W1*(X0+X1+...+Xn)
                # 对W0求偏导就可以得到W0的梯度值
    loss /= num_train  # 求平均loss
    loss += reg * np.sum(W * W)  # 加入正则项对模型进行惩罚
    # 笔记1:
    # 正则化系数越大对模型的惩罚就越高,模型就会越复杂,泛化能力就越弱
    dW /= num_train
    dW += reg * 2 * W
    # Q1:为什么dW的正则项是 2 * W？
    # A1:因为在计算梯度时,对于偏置b并没有进行正则化,所以在计算dw的正则化值时,只需要考虑权重w的正则化项即可.
    # 而在计算梯度时,需要对这个正则项求导数,由于平方项求导后系数为2,所以在计算正则项的梯度时,需要乘以2,得到的结果就是2 * W
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    函数功能: 计算损失值和梯度矩阵(向量化计算)

    :param W: 权重矩阵(M*C的二维矩阵: M表示样本有多少个特征, C表示数据集有多少分类)
    :param X: 训练集(N*M的二维矩阵: N表示有多少样本, M表示样本有多少个特征)
    :param y: 训练集的标签(1*N的二维矩阵)
    :param reg: 正则化系数
    :return: 损失值和梯度矩阵
    """
    loss = 0.0
    dW = np.zeros(W.shape)
    """ 一次计算所有数据的分数矩阵 """
    scores = X.dot(W)
    """ 正确分类的分数 """
    correct_scores = scores[np.arange(scores.shape[0]), y].reshape(-1, 1)
    # 笔记1:
    # scores[np.arange(scores.shape[0]), y]能选中训练数据的正确分类的分数
    """ 获取各个训练样本的损失 """
    margins = np.maximum(0, scores - correct_scores + 1)
    # 笔记2:
    # 通过广播,一次计算出所有margins
    margins[np.arange(margins.shape[0]), y] = 0
    # 笔记3:
    # 将每个训练数据的正确分类的margin设置为0
    """ 计算整体的平均损失 """
    loss = np.sum(margins) / X.shape[0] + reg * np.sum(W * W)
    binary = margins
    binary[margins > 0] = 1
    # 笔记4:
    # 计算所有非零的margin,将对应的binary数组中的元素置为1,表示这些样本会对损失函数产生梯度
    row_sum = np.sum(binary, axis=1)
    # 笔记5:
    # 计算binary数组每一行的和,即表示每个样本对应的梯度值需要被减去的次数
    binary[np.arange(X.shape[0]), y] -= row_sum
    # 笔记6:
    # binary[np.arange(X.shape[0]), y]表示在binary数组中选取每个样本的正确类别的所在位置(列)
    # 笔记7:
    # 假设loss1 = W0(c1*X1 + c2*X2 + ... + ci*Xi), binary矩阵存储的就是c1,c2,...ci的值
    dW = X.T.dot(binary) / X.shape[0] + reg * 2 * W
    return loss, dW
