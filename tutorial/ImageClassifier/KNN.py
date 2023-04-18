# KNN分类器

import numpy as np

class Classifier(object):
    def __init__(self):
        self.y_train = None
        self.X_train = None

    def train(self, X, y):
        """
        将"训练集"和"标签"保存下来以便预测的时候做对比

        :param X: 训练集(M * N 的二维矩阵),每一行都是一个训练样例
        :param y: 标签(1 * M的一维矩阵),一个训练样例对应的一个标签
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X_test, k, num_loops):
        """
        对测试集进行预测

        :param X_test: 测试集(M * N的二维矩阵),每一行都是一个测试用例
        :param k: 投票的"邻居"个数
        :param num_loops: 选择你要用的方法
        :return: 返回一个ndarray(预测结果)
        """
        if num_loops == 0:
            dists = self.compute_distance_no_loops(X_test)
        elif num_loops == 1:
            dists = self.compute_distance_one_loops(X_test)
        else:
            dists = self.compute_distance_two_loops(X_test)
        y_pred = self.predict_label(dists, k)
        return y_pred

    def compute_distance_no_loops(self, X_test):
        print("compute_distance_no_loops")
        L2_dists = np.sum(X_test ** 2, axis=1).reshape(-1, 1) + np.sum(self.X_train ** 2, axis=1) - 2 * np.dot(X_test, self.X_train.T)
        print(L2_dists[0])
        print(L2_dists.shape)
        return L2_dists

    def compute_distance_one_loops(self, X_test):
        """
        使用一层"for循环"来计算"测试数据"到各个"训练数据"的欧氏距离(L2_Distance)

        :param X_test: 测试集
        :return: 返回一个ndarray(所有"测试数据"各自到"训练数据"的欧式距离)
        """
        print("compute_distance_one_loops")
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        L2_dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            sub_distance_array = np.abs(X_test[i, :] - self.X_train)
            L2_dists[i] = np.sum(sub_distance_array**2, axis=1)
        print(L2_dists[0])
        return L2_dists

    def compute_distance_two_loops(self, X_test):
        print("compute_distance_two_loops")
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        L2_dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                distance = np.sum((X_test[i] - self.X_train[j])**2)
                L2_dists[i][j] = distance
        print(L2_dists[0])
        return L2_dists

    def predict_label(self, dists, k):
        """
        选取离测试数据"欧式距离"最近的k个点,来进行投票,选取票数最多的作为最终预测结果

        :param dists: 测试集与数据集之间的欧氏距离
        :param k: 要选取的邻居个数
        :return: 返回预测的结果
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            closest_y = np.array(closest_y, dtype='i8')
            sub_dists = dists[i]
            sort_sub_dists = np.argsort(sub_dists)
            for j in range(k):
                closest_y = np.append(closest_y, self.y_train[sort_sub_dists[j]])
            if k > 1:
                counts = np.bincount(closest_y)  # 获取每个元素出现的次数
                predict = np.argmax(counts)  # 获取出现次数最多的值
                y_pred[i] = predict
            else:
                y_pred[i] = closest_y[0]
        return y_pred
