import matplotlib.pyplot as plt
import pickle
import numpy as np
from tutorial.ImageClassifier.Batch_Gradient_Descent import Classifier

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
print(X_train[0])
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


