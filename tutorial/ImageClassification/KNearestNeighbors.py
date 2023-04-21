import matplotlib.pyplot as plt
import pickle
import numpy as np
from tutorial.ImageClassifier.KNN import Classifier

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

""" 将部分图片数据集可视化 """
# plt.imshow(X_train[1])
# plt.title('Label: {}'.format(y_train[1]))
# plt.show()
#
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# num_classes = len(classes)
# samples_per_class = 7
# for y, cls in enumerate(classes):
#     # 笔记1: enumerate(classes)可以将y, cls组成键值对(y:cls)==>(0:plane)(1:car)(2:bird)....
#     idxs = np.flatnonzero(y_train == y)
#     # 笔记2: (y_train==y) 可以把"标签相同的数据"的下标提取出来,其实就是分类(相同标签的数据在一个列表中)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         # 笔记3: subplot()是用于在一张图中绘制多个子图的
#         plt.imshow(X_train[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()

""" 削减数据集的规模,减少训练时间 """
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]
X_train = np.reshape(X_train, (X_train.shape[0], -1))  # 训练集
X_test = np.reshape(X_test, (X_test.shape[0], -1))  # 测试集

""" K-Fold 交叉验证 """
num_folds = 5
k_choices = [1, 3, 5, 8, 10]
X_train_folds = np.split(X_train, indices_or_sections=num_folds, axis=0)
y_train_folds = np.split(y_train, indices_or_sections=num_folds)

k_to_accuracies = {}
for i in range(len(k_choices)):
    k = k_choices[i]
    accuracies = []
    for j in range(num_folds):
        """ 将数据集分为训练集和验证集 """
        X_validate_test = X_train_folds[j]  # 验证集
        y_validate_test = y_train_folds[j]  # 验证集对应的标签
        if j == 0:
            X_validate_train = np.array(X_train_folds[j + 1:])
            y_validate_train = np.array(y_train_folds[j + 1:])
        elif (j > 0) & (j < num_folds - 1):
            X_validate_train = np.append(np.array(X_train_folds[:j]), np.array(X_train_folds[j + 1:]), axis=0)
            y_validate_train = np.append(np.array(y_train_folds[:j]), np.array(y_train_folds[j + 1:]), axis=0)
        else:
            X_validate_train = np.array(X_train_folds[:j])
            y_validate_train = np.array(y_train_folds[:j])
        X_validate_train = np.reshape(X_validate_train, (-1, X_validate_train.shape[2]))  # 训练集
        y_validate_train = np.reshape(y_validate_train, (-1,))  # 训练集对应的标签

        """ 训练模型 """
        KNN = Classifier()
        KNN.train(X_validate_train, y_validate_train)

        """ 对验证集进行预测,同时调整超参数 """
        predict_dists = KNN.predict(X_validate_test, k, 1)

        """ 评估预测结果 """
        num_correct = np.sum(predict_dists == y_validate_test)
        accuracies.append(float(num_correct) / 1000)
    k_to_accuracies[k] = accuracies

print(k_to_accuracies)

""" 绘制不同K值下的"准确率的散点图","准确率的标准差","平均准确率" """
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)
    # 笔记1:
    # scatter函数是用来绘制散点图的
    # 第一个参数是x轴数据,第二个参数是y轴数据,因此我们将k作为x轴数据,accuracies(准确率列表)作为y轴数据,绘制成点的集合的的形式
    # 笔记2:
    # 在Python中,*符号除了表示乘法以外,还可以用于复制序列中的元素.
    # 因此,[k] * len(accuracies)的含义是创建一个长度为len(accuracies)的列表,其中所有元素都是k.
    # 例如,如果k=1,len(accuracies)=3,则[k] * len(accuracies)将生成[1, 1, 1]这样的列表.

accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
# 笔记1:
# np.mean(v)计算了value(即准确率列表)的平均值,最终返回一个平均值列表,按照k值的顺序排列
# k代表k值,v代表对应k值下的准确率列表.其中,k_to_accuracies是一个字典类型,key为k值,value为准确率列表v.
# 因此,sorted(k_to_accuracies.items())会将字典按照key进行排序,返回一个由键值对元组组成的列表,列表中每个元素都是(key, value)的形式.

accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
# 笔记1:
# np.std(v)的作用是计算不同k值下的准确率列表的标准差
# 笔记2:
# 标准差就是"方差"的算术平方根

plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
# 笔记1:
# plt.errorbar会在给定的k值处,绘制出每个k值下的所有准确率,并在每个准确率上方和下方绘制误差条,误差条的长度表示对应准确率的标准差.

plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

""" 将模型的参数调整到最佳状态来预测'测试数据' """
best_k = 1  # 通过绘制出来的图像找到平均准确率的参数k
classifier = Classifier()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k, num_loops=1)
# 笔记1:
# 如果用"完全向量化"的算法处理矩阵,那么内存开销会非常大,所以本次预测全程使用"部分向量化"的算法

num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print(accuracy)

""" 对KNN的三种方法进行效率上的评估 """
# two_loop_time = KNN.time_function(KNN.compute_distance_two_loops, X_test)
# print('Two loop version took %f seconds' % two_loop_time)  # 运行时间==> 30.367281 seconds
#
# one_loop_time = KNN.time_function(KNN.compute_distance_one_loops, X_test)
# print('One loop version took %f seconds' % one_loop_time)  # 运行时间==> 20.551801 seconds
#
# no_loop_time = KNN.time_function(KNN.compute_distance_no_loops, X_test)
# print('No loop version took %f seconds' % no_loop_time)  # 运行时间==> 16 seconds
