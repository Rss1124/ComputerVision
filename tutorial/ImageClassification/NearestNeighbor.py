# 最近邻分类器（Nearest Neighbor Classifier）是一种基于距离度量的分类器
# 主要的思想: 将测试数据的每一个"测试样例"依次拿去与"训练集的样例"一一比较(曼哈顿距离),将"测试样例"分类到曼哈顿距离最近的"训练样例"

import numpy as np


class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        """
        :param X: 训练集(M * N 的二维矩阵),每一行都是一个训练样例
        :param y: 标签(1 * M的一维矩阵),一个训练样例对应的一个标签
        """
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """
        :param X: 测试集(M * N 的二维矩阵),每一行都是一个测试样例
        :return: 返回测试集的标签集合
        """
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        for i in range(num_test):
            # 通过"广播"的方式让每个"测试样例"与"训练集"进行对比
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            # "测试样例"离哪条"训练数据"最近,返回下标
            min_index = np.argmin(distances)
            # 给"测试样例"打上"标签"
            Ypred[i] = self.ytr[min_index]
        return Ypred


# 实例:
Xtr = np.array([[1, 3, 5], [9, 10, 6], [8, 9, 7]])
print("训练集如下: ")
ytr = np.array(["标签1", "标签2", "标签3"])
print("样本  特征值1  特征值2  特征值3  标签")
print("1     1       3       5       标签1")
print("2     9       10      6       标签2")
print("3     8       9       7       标签3")
print(" ")

print("测试集如下:")
print("样本  特征值1  特征值2  特征值3")
print("1     1       2       3     ")
print("2     4       5       6     ")
print("3     7       8       9     ")
test_set = np.array([[1, 2, 3], [4, 5, 6, ], [7, 8, 9]])
test = NearestNeighbor()
test.train(Xtr, ytr)
test_arr = test.predict(test_set)
print(" ")

print("测试集到数据集的曼哈顿距离:")
print("测试样本/训练样本      样本1  样本2  样本3")
print("     样本1             3     19     18")
print("     样本2             6     10     9")
print("     样本3             15    7      4")
print("")

print("测试集的预测结果如下:")
print("样本   标签")
print("1     " + str(test_arr[0]))
print("2     " + str(test_arr[1]))
print("3     " + str(test_arr[2]))
