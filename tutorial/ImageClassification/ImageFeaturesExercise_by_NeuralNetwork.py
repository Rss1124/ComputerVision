import numpy as np
from matplotlib import pyplot as plt

from BaseUtils.data_utils import get_CIFAR10_data
from BaseUtils.features_utils import hog_feature, color_histogram_hsv, extract_features
from BaseUtils.vis_utils import visualize_grid
from tutorial.NeuralNetwork.fully_connected import Layer
from tutorial.NeuralNetwork.solver import Solver

X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
X_train = X_train.transpose(0, 2, 3, 1)
X_val = X_val.transpose(0, 2, 3, 1)
X_test = X_test.transpose(0, 2, 3, 1)
X_dev = X_dev.transpose(0, 2, 3, 1)

""" 初始化"特征函数"数组 """
# feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
feature_fns = [hog_feature, color_histogram_hsv]

""" 对数据集使用不同特征函数进行特征处理 """
print("-------------------------------------- 特征工程进行中 --------------------------------------")
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

""" 对使用这些新的特征之前,再进行一次归一化,来提高特征的表达能力 """
# 笔记1:
# 在这里采用了Z-score标准化,即先减去平均值,再除以标准差,这种归一化方法可以使数据的分布更接近正态分布
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)  # 平均值
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat
std_feat = np.std(X_train_feats, axis=0, keepdims=True)  # 标准差
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

print(X_train_feats.shape)

""" 初始化两层神经网络模型 """
input_dim = X_train_feats.shape[1]
hidden_dim = 250
num_classes = 10

""" 初始化解决器 """

data = {}
data.setdefault("X_train", X_train_feats)
data.setdefault("y_train", y_train)
data.setdefault("X_val", X_val_feats)
data.setdefault("y_val", y_val)
data.setdefault("X_test", X_test_feats)
data.setdefault("y_test", y_test)

learning_rates = [2e-2, 5e-3, 1e-3]
lr_decays = [1, 0.96, 0.92]
best_val_acc = -999
best_model = None

""" 交叉验证 """

print("-------------------------------------- 模型训练进行中 --------------------------------------")
for lr in learning_rates:
    for lr_decay in lr_decays:
        model = Layer(input_dim, hidden_dim, num_classes)
        params = (lr, lr_decay)
        solver = Solver(model, data,
                        optim_config={"learning_rate": lr},
                        num_epochs=20, batch_size=500, print_every=98, lr_decay=lr_decay)

        """ 每次训练会得到一个当前最佳的model和当前最佳的val_acc """
        solver.train()
        model_temp, best_val_acc_temp = solver.model, solver.best_val_acc
        """ 将loss以及accuracy还有权重矩阵可视化 """
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.subplot(2, 1, 1)
        plt.title("(步长: " + str(params[0]) + ")" + "(步长衰退率: " + str(params[1]) + ")")
        plt.plot(solver.loss_history, 'o')
        plt.xlabel('Iteration')

        plt.subplot(2, 1, 2)
        plt.title("(步长: " + str(params[0]) + ")" + "(步长衰退率: " + str(params[1]) + ")")
        plt.plot(solver.train_acc_history, '-o', label='train')
        plt.plot(solver.val_acc_history, '-o', label='val')
        plt.plot([0.5] * len(solver.val_acc_history), 'k--')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.gcf().set_size_inches(15, 12)
        plt.show()

        if best_val_acc_temp > best_val_acc:
            best_val_acc = best_val_acc_temp
            best_model = model_temp

y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
test_acc = (y_test_pred == data['y_test']).mean()
print('final test set accuracy: %f' % test_acc)
