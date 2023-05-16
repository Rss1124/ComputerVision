import numpy as np
from matplotlib import pyplot as plt

from BaseUtils.data_utils import get_CIFAR10_data
from BaseUtils.gradient_check import grad_check_sparse
from tutorial.ImageClassifier.SVM.Linear_Classifier import Classifier
from tutorial.ImageClassifier.SVM.softmax import softmax_loss_vectorized

""" 获取已经处理好的数据集 """
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()

""" 重新处理数据集的shape """
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
X_dev = X_dev.reshape(X_dev.shape[0], -1)

""" 给数据集添加偏置维度 """
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

""" 随机生成一个权重矩阵 """
W = np.random.randn(3073, 10) * 0.0001
loss, grad = softmax_loss_vectorized(W, X_dev, y_dev, 0.0)

""" 作为一个粗略的完整性检查，我们的损失应该接近于-log(0.1) """
print("-----------------检查数据集的分类情况是否平衡-----------------")
print('loss: %f' % loss)
print('sanity check: %f' % (-np.log(0.1)))
# Question1:
# 为什么要接近与-log(0.1)?
# Answer:
# 首先假设我们有10个分类,在训练模型之前,我们要先验证数据集的分类情况是否平衡
# 如果数据集的分类是平衡的,那么就意味着每个分类的概率为1/10,那么它的loss就是2.302585
# 数据集的分类越平衡,最后训练出来的效果就越好

""" 使用数值梯度去检查我们计算的解析梯度是否正确 """
print("-----------------验证梯度矩阵的正确性----------------")
f = lambda w: softmax_loss_vectorized(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)

""" 普通算法和向量化算法的性能比较 """
# tic = time.time()
# loss_naive, grad_naive = softmax_loss_naive(W=W, X=X_dev, y=y_dev, reg=0.1)
# toc = time.time()
# print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))  # ==> 66.017381s
#
# tic = time.time()
# loss_vectorized, _ = softmax_loss_vectorized(W=W, X=X_dev, y=y_dev, reg=0.1)
# toc = time.time()
# print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))  # ==> 0.006006s
# print('difference: %f' % (loss_naive - loss_vectorized))  # ==> 0.000000

""" 使用验证集来调整超参数(步长和正则化系数) """
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
        loss_hist = svm_val.train(f=softmax_loss_vectorized, X=X_train, y=y_train, learning_rate=j,
                                  reg=i, num_iters=400, batch_size=1000, verbose=False)
        """ 评估模型准确率 """
        y_train_pred = svm_val.predict(X_train)
        y_train_accuracy = np.mean(y_train == y_train_pred)
        y_val_pred = svm_val.predict(X_val)
        y_val_accuracy = np.mean(y_val == y_val_pred)
        accuracy = (y_train_accuracy, y_val_accuracy)
        """ 找寻当前最佳的分类器 """
        if y_val_accuracy > best_accuracy:
            best_accuracy = y_val_accuracy
            best_svm = svm_val
        results[params] = accuracy
        """ 绘制出不同正则化系数和步长的情况下,loss的变化 """
        plt.plot(loss_hist)
        plt.xlabel('Iteration number')
        plt.ylabel('Loss value')
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.title("(正则化系数: " + str(params[0]) + ")" + "(步长: " + str(params[1]) + ")")
        plt.show()

# """ 不同超参数和准确率的汇总 """
# for lr, reg in sorted(results):
#     train_accuracy, val_accuracy = results[(lr, reg)]
#     print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
#         lr, reg, train_accuracy, val_accuracy))
# print('best validation accuracy : %f' % best_accuracy)

""" 对最后的测试集进行预测 """
print("-----------------使用最佳线性分类器对测试集进行预测-----------------")
y_test_pred = best_svm.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('final test set accuracy: %f' % test_accuracy)

""" 可视化权重矩阵 """
w = best_svm.W[:-1, :]
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)
    """ 将每个权重矩阵的值扩大至0-255 """
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])
plt.suptitle("权重矩阵可视化")
plt.show()



