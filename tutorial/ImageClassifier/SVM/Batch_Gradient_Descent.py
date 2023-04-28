import numpy as np


def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape)
    """ 计算整体的平均损失值 """
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                """ 计算损失 """
                loss += margin
                """ 计算梯度 """
                dW[:, j] += X[i].T
                dW[:, y[i]] -= X[i].T
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += reg * 2 * W
    # Q1:为什么dW的正则项是 2 * W？
    # A1:因为在计算梯度时,对于偏置b并没有进行正则化,所以在计算dw的正则化值时,只需要考虑权重w的正则化项即可.
    # 而在计算梯度时,需要对这个正则项求导数,由于平方项求导后系数为2,所以在计算正则项的梯度时,需要乘以2,得到的结果就是2 * W
    return loss, dW

