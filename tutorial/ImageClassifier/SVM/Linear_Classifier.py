import numpy as np

from tutorial.ImageClassifier.SVM.Multi_HingeLoss import svm_loss_vectorized


class Classifier(object):
    def __init__(self):
        self.W = None

    def train(
            self,
            f,
            X,
            y,
            learning_rate=0.001,
            reg=0.00001,
            num_iters=100,
            batch_size=200,
            verbose=False,
    ):
        """

        :param f: 线性分类器所用的损失函数名称
        :param X: 训练集(N*M的二维矩阵: N表示有多少样本, M表示样本有多少个特征)
        :param y: 训练集的标签(1*N的二维矩阵)
        :param learning_rate: 每次进行梯度下降的步长
        :param reg: 正则化系数
        :param num_iters: 模型进行迭代的次数
        :param batch_size: 参与"权重矩阵"调整的样本数量
        :param verbose: 是否显示进度条
        :return: 损失集(1*num_iters的二维矩阵)
        """
        num_train, dim = X.shape
        # 笔记1:
        # X.shape有两个值, num_train和dim会分别获取X.shape[0]与X.shape[1]
        num_classes = np.max(y) + 1
        """ 随机初始化一个权重矩阵(M*C的二维矩阵) """
        if self.W is None:
            self.W = np.random.randn(dim, num_classes) * 0.001
        """ 随机梯度下降 """
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None
            mask = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[mask]
            y_batch = y[mask]
            loss, grad = f(self.W, X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W -= grad * learning_rate
            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))
        return loss_history

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

