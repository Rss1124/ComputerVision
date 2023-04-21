# KNN分类器

import numpy as np


class Classifier(object):
    def __init__(self):
        self.y_train = None
        self.X_train = None

    def train(self, X, y):
        """
        函数功能: 将"训练集"和"标签"保存到对象中

        :param X: 训练集(M * N 的二维矩阵)
        :param y: 标签集(1 * M的一维矩阵)
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X_test, k, num_loops):
        """
        函数功能: 对测试集进行预测 \n
        函数特点: 通过K个离"测试数据"最近的点进行投票,提高准确率

        :param X_test: 测试集(M * N的二维矩阵)
        :param k: 投票的"邻居"个数
        :param num_loops: 选择你要用的方法
        :return: 返回一个ndarray(预测结果)
        """
        if num_loops == 0:
            dists = self.compute_distance_no_loops(X_test)
        elif num_loops == 1:
            dists = self.compute_distance_one_loops(X_test, k)
        else:
            dists = self.compute_distance_two_loops(X_test)
        y_pred = self.predict_label(dists, k)
        return y_pred

    def compute_distance_no_loops(self, X_test):
        """
        函数功能: 计算"测试数据"到各个"训练数据"的欧氏距离(L2_Distance) \n
        函数特点: 完全使用向量化计算(从而避免任何显性for循环)

        :param X_test: 测试集(M * N的二维矩阵)
        :return: 返回一个ndarray(包含所有"测试数据"各自到"训练数据"的欧式距离)
        """
        # L2_dists = np.sum(X_test**2, axis=1).reshape(-1, 1) + \
        #            np.sum(self.X_train**2, axis=1) - \
        #            2 * np.dot(X_test, self.X_train.T).astype(np.float64)
        L2_dists = np.sum((X_test[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2, axis=2)
        # Problems:
        # 算法1:np.sum(X_test**2, axis=1).reshape(-1, 1)+np.sum(self.X_train**2, axis=1)-2*np.dot(X_test, self.X_train.T)
        # 算法2:np.sum((X_test[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2, axis=2)
        # 虽然两者都是计算L2距离,但是两者最终的距离矩阵可能会不一样,甚至有时候差距会非常大

        # Answer:
        # 对于算法1,可能会存在浮点数精度的问题,因为在计算过程中有很多的乘除操作特别是平方操作,如果操作不当就可能导致精度损失.
        # 而算法2是直接利用numpy内置的函数,减少了加减乘除操作.避免了浮点数精度的问题. 如果在数据量非常大的情况下,精度损失会更加大.
        # poi: 所以向量化计算尽量用方法二!

        # 可以使用下列代码进行验证:
        # X_test = np.random.rand(100, 10)
        # X_train = np.random.rand(1000, 10)
        #
        # L2_dists_1 = np.sum(X_test ** 2, axis=1).reshape(-1, 1) + np.sum(X_train ** 2, axis=1) - 2 * np.dot(X_test,
        #                                                                                                     X_train.T)
        # L2_dists_2 = np.sum((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2, axis=2)
        #
        # for i in range(L2_dists_1.shape[0]):
        #     if L2_dists_1[0][i] != L2_dists_2[0][i]:
        #         print(i)
        #         print(L2_dists_1[0][i])
        #         print(L2_dists_2[0][i])
        #         print("not equal")
        #         break
        # print("finish")
        return L2_dists

    def compute_distance_one_loops(self, X_test, k):
        """
        函数功能: 计算"测试数据"到各个"训练数据"的欧氏距离(L2_Distance) \n
        函数特点: 使用部分向量化计算(将for循环降低到一层)

        :param X_test: 测试集(M * N的二维矩阵)
        :return: 返回一个ndarray(包含所有"测试数据"各自到"训练数据"的欧式距离)
        """
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        L2_dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            print("k=" + str(k) + "==>" + str(i) + '/' + str(num_test))
            sub_distance_array = np.abs(X_test[i, :] - self.X_train)
            L2_dists[i] = np.sum(sub_distance_array ** 2, axis=1)
        return L2_dists

    def compute_distance_two_loops(self, X_test):
        """
        函数功能: 计算"测试数据"到各个"训练数据"的欧氏距离(L2_Distance) \n
        函数特点: 使用两层for循环依次算出每个距离

        :param X_test: 测试集(M * N的二维矩阵)
        :return: 返回一个ndarray(包含所有"测试数据"各自到"训练数据"的欧式距离)
        """
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        L2_dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            # print(str(i) + '/' + str(num_test))
            for j in range(num_train):
                distance = np.sum((X_test[i] - self.X_train[j]) ** 2)
                L2_dists[i][j] = distance
        return L2_dists

    def predict_label(self, dists, k):
        """
        函数功能: 对测试集进行预测 \n
        函数特点: 通过K个离"测试数据"最近的点来进行投票,提高准确率

        :param dists: 欧式距离矩阵
        :param k: 投票的人数
        :return: 返回一个ndarray数组(标签数组)
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

    @staticmethod
    def time_function(f, *args):
        """
        函数功能: 用来检测函数的运行时间

        :param f: 待检测的函数
        :param args: 函数所需要的参数
        :return: 具体的时间
        """
        import time
        tic = time.time()
        f(*args)
        toc = time.time()
        return toc - tic
