import math
import time

import matplotlib.pyplot as plt
import pickle
import numpy as np
from tutorial.ImageClassifier.SVM.Batch_Gradient_Descent import svm_loss_naive, svm_loss_vectorized
from BaseUtils.gradient_check import grad_check_sparse
from tutorial.ImageClassifier.SVM.Linear_Classifier import Classifier

train_batch_url = "../DataSet/cifar-10-batches-py/data_batch_1"
test_batch_url = "../DataSet/cifar-10-batches-py/test_batch"


def get_train_data_from_cifar10(file_url):
    with open(file_url, 'rb') as f:
        data_train_batch = pickle.load(f, encoding='bytes')
    return data_train_batch


def get_test_data_from_cifar10(file_url):
    with open(file_url, 'rb') as f:
        data_test_batch = pickle.load(f, encoding='bytes')
    return data_test_batch


data_batch = get_train_data_from_cifar10(train_batch_url)
test_batch = get_test_data_from_cifar10(test_batch_url)

""" 从buffer流中提取数据信息和标签信息 """
X_train = data_batch[b'data']
y_train = np.array(data_batch[b'labels'])
X_test = test_batch[b'data']
y_test = np.array(test_batch[b'labels'])

""" 将部分图片数据集可视化 """
X_train = X_train.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

""" 将数据集分为训练集,验证集,测试集 """
X_train = X_train.reshape(X_train.shape[0], -1)
num_training = 9000
num_validation = 1000
num_test = 1000
num_dev = 500  # dev是开发集,是训练集的子集(只有500条数据),方便做测试

mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]  # 验证集_数据
y_val = y_train[mask]  # 验证集_标签
# print("验证集_数据:" + str(X_val.shape))
# print("验证集_标签:" + str(y_val.shape))
mask = range(num_training)
X_train = X_train[mask]  # 训练集_数据
y_train = y_train[mask]  # 训练集_标签
# print("训练集_数据:" + str(X_train.shape))
# print("训练集_标签:" + str(y_train.shape))
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]  # 开发集_数据
y_dev = y_train[mask]  # 开发集_标签
# print("开发集_数据:" + str(X_dev.shape))
# print("开发集_标签:" + str(y_dev.shape))
mask = range(num_test)
X_test = X_test[mask]  # 测试集_数据
y_test = y_test[mask]  # 测试集_标签
# print("测试集_数据:" + str(X_test.shape))
# print("测试集_标签:" + str(y_test.shape))

""" 对数据集进行预处理 """
mean_image = np.mean(X_train, axis=0)
# 笔记1:
# 这里的mean_image有3072个数值(每个图片都是3*32*32的),每个数值代表所有图片的某个像素块的平均值
X_train = X_train - mean_image
X_val = X_val - mean_image
X_test = X_test - mean_image
X_dev = X_dev - mean_image
# 笔记2:
# 对整个数据集进行了'均值归一化'操作,即减去均值,通过这种方式可以使数据中心化.
# 即所有特征的数值都接近于0,这样做可以减少数据中的冗余信息,使得模型更加专注于真正有用的特征,同时可以加速模型的收敛速度.
# Question1:
# 如果让训练集的数据减去训练集的均值,验证集的数据减去验证集的均值,测试集的数据减去测试集的均值.会对性能有影响吗？
# print(X_train[0])
# Question2:
# 该案例的预预处理只减去了均值,但是不同特征之间的差距还是非常大,这会模型造成影响.
# 如果再除以它的标准化(进一步归一化),会对性能有影响吗?
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
# 笔记3:
# 此处是给每个样本的特征向量中加上一个额外的偏置维度(偏执项数值为1)
# 假设我们有一个权重矩阵W,一个样本的特征向量xi,那么我们就可以得到一个公式: f(x) = w1.x1 + ... + wi.xi
# 现在我们给W加上一个偏置维度,那么我们的公式就变成了: f(x) = w1.x1 + ... + wi.xi + b(偏置值)
# SVM的训练过程就是不断调整W和b,使得模型在未见过的测试数据上能够具有很好的泛化能力.

""" 随机生成一个权值矩阵 """
W = np.random.randn(3073, 10) * 0.0001

""" 使用开发集训练一次模型,验证梯度下降算法并获取"损失值"和"梯度矩阵" """
print("-----------------batch梯度下降-----------------")
loss, grad = svm_loss_naive(W, X_dev, y_dev, 5e1)
f = lambda w: svm_loss_naive(w, X_dev, y_dev, 5e1)[0]
print("loss:", loss)
# 笔记1:
# lambda是一个用于定义匿名函数的关键字.例子如下:
# # 定义普通函数
# def square(x):
#     return x**2
#
# # 定义匿名函数
# f = lambda x: x**2
#
# # 使用函数
# print(square(5))  # 输出: 25
# print(f(5))  # 输出: 25

# 笔记2:
# 此处f的作用是获取损失值

""" 检查梯度矩阵的正确性 """
print("-----------------验证梯度矩阵的正确性----------------")
grad_numerical = grad_check_sparse(f, W, grad)
# 笔记1:
# grad_check_sparse 函数会使用数值梯度和解析梯度之间的相对误差(也称为相对误差)来检查梯度的正确性.
# 具体而言,它会使用数值梯度(使用一小部分样本通过'导数的定义式'计算得出)和解析梯度('根据loss的计算公式直接求导')来计算相对误差.
# 如果相对误差低于某个阈值,则认为梯度计算是正确的.
# Question3:
# 在一段时间内，一个维度在gradcheck将不完全匹配。这种差异是由什么引起的呢?
# Question4:
# 这是一个值得关注的理由吗?
# Question5:
# 在一维中，梯度检查可能失败的简单例子是什么?
# Question6:
# 频繁发生这种情况的话边际效应会如何改变?

""" 评估损失函数(传统方法)和损失函数(向量化计算)的时间效率 """
# tic = time.time()
# loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
# toc = time.time()
# print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))  # ==> 0.062057s
#
# tic = time.time()
# loss_vectorized, _ = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
# toc = time.time()
# print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))  # ==> 0.004004s
# print('difference: %f' % (loss_naive - loss_vectorized))  # ==> 0.000000

print("-----------------随机梯度下降-----------------")
svm = Classifier()

""" 评估随机梯度下降的时间效率 """
tic = time.time()
loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4,
                      num_iters=1500, verbose=True)
toc = time.time()
print('That took %fs' % (toc - tic))

""" 绘制出迭代次数和loss的关系图 """
# plt.plot(loss_hist)
# plt.xlabel('Iteration number')
# plt.ylabel('Loss value')
# plt.show()

""" 评估线性分类模型的准确率 """
y_train_pred = svm.predict(X_train)
print('training accuracy: %f' % (np.mean(y_train == y_train_pred),))
y_val_pred = svm.predict(X_val)
print('validation accuracy: %f' % (np.mean(y_val == y_val_pred),))

""" 使用交叉验证调整超参数 """
print("-----------------观察不同正则化值和步长的情况下,loss的变化,并找到当前最佳的svm线性分类器-----------------")
k_regs = [1e-4, 1e5, 2.5e4, 5e-5]
k_learning_rates = [1e-4, 1e3, 1e-7, 5e4]
results = {}
best_accuracy = -1
best_svm = None
for i in k_regs:
    for j in k_learning_rates:
        params = (i, j)
        """ 训练模型 """
        svm_val = Classifier()
        loss_hist = svm_val.train(X=X_train, y=y_train, learning_rate=j,
                                  reg=i, num_iters=400, batch_size=200, verbose=False)
        """ 评估模型准确率 """
        y_train_pred = svm_val.predict(X_train)
        y_train_accuracy = np.mean(y_train == y_train_pred)
        y_val_pred = svm_val.predict(X_val)
        y_val_accuracy = np.mean(y_val == y_val_pred)
        accuracy = (y_train_accuracy, y_val_accuracy)
        """ 找寻当前最佳的分类器 """
        if y_train_accuracy > best_accuracy:
            best_accuracy = y_train_accuracy
            best_svm = svm_val
        results[params] = accuracy
        """ 绘制出不同正则化系数和步长的情况下,loss的变化 """
        plt.plot(loss_hist)
        plt.xlabel('Iteration number')
        plt.ylabel('Loss value')
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.title("(正则化系数: " + str(params[0]) + ")" + "(步长: " + str(params[1]) + ")")
        plt.show()

""" 不同超参数和准确率的汇总 """
# for reg, lr in sorted(results):
#     train_accuracy, val_accuracy = results[(reg, lr)]
#     print('lr: %e reg: %e train accuracy: %f val accuracy: %f' % (
#         lr, reg, train_accuracy, val_accuracy))

""" 对最后的测试集进行预测 """
print("-----------------使用获取到的最佳svm线性分类器对测试集进行预测-----------------")
y_test_pred = best_svm.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('final test set accuracy: %f' % test_accuracy)

""" 可视化权值矩阵 """
w = best_svm.W[:-1, :]
# 笔记1:
# W[:-1, :] == W[:W.shape[0]-1, :]
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    # 笔记2:
    # squeeze()函数将长度为1的维度去除,这是因为当对一个数组进行切片操作时,有时会产生长度为1的维度,可能会影响代码的处理效率.例如:
    # A = np.array([[1,2,3], [4,5,6]])
    # A_slice = A[0:1, :]
    # 得到的A_slice的shape是(1,3),我们可以看到此时是一个二维矩阵.而进行squeeze()之后,shape就变为了(3,)一个一维矩阵
    # 从表示上看都是一样,但是后者在计算机的处理中可能效率会更高一点
    # 笔记3:
    # (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)将所有权重值减去最小值并除以最大值和最小值的差,将结果缩放到0和1之间.
    # 最后,255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)将结果乘以255,将结果缩放到0和255之间.以便可视化
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])
plt.suptitle("权值矩阵可视化")
plt.show()
