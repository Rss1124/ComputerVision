import matplotlib.pyplot as plt
import pickle
import numpy as np

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

""" 处理训练数据集和测试数据集 """
X_train = X_train.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
X_test = X_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
# Q1: 为什么会将图片数据reshape为4维矩阵(10000, 3, 32, 32)？
# A1: 参数解释 "10000": 图像的个数, "3": 颜色通道个数(RGB), "32": CIFAR-10数据集中的图像都是32*32像素的
#     在CIFAR-10数据集中的图像是按照行优先顺序存储的, 所以每个图像就是一个3072的字节流, 前1024个是红色通道, 中间1024个是绿色通道,
#     最后1024个是蓝色通道
# Q2: 如何理解transpose(0, 2, 3, 1)？
# A2: 根据reshape操作,我们可以发现"通道数"在第二个维度上,但在卷积层中通常会把"通道数"放在最后一个维度(所以该操作就是做这个的)
# Q3: 那我可以直接reshape成(10000, 32, 32, 3)吗？
# A3: 不可以, 这样会导致"该颜色通道"会混杂其他的颜色.
#
# 如果不好理解的话,请尝试以下代码:
# arr = np.arange(24).reshape(4, 3, 2)
# print(arr)
# print("")
# arr = arr.transpose(0, 2, 1)
# print(arr)
print(X_train[1])

""" 将部分图片数据集可视化 """
plt.imshow(X_train[1])
plt.title('Label: {}'.format(y_train[1]))
plt.show()

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    # 笔记1: enumerate(classes)可以将y, cls组成键值对(y:cls)===>(0:plane)(1:car)(2:bird)....
    idxs = np.flatnonzero(y_train == y)
    # 笔记2: (y_train==y) 可以把"标签相同的数据"的下标提取出来,其实就是分类(相同标签的数据在一个列表中)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        # 笔记3: subplot()是用于在一张图中绘制多个子图的
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

