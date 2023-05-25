from matplotlib import pyplot as plt

from BaseUtils.data_utils import get_CIFAR10_data
from BaseUtils.vis_utils import visualize_grid
from tutorial.NeuralNetwork.fully_connected import Layer
from tutorial.NeuralNetwork.solver import Solver

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

""" 获取训练集,验证集,测试集和开发集"""
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
X_dev = X_dev.reshape(X_dev.shape[0], -1)
data = {}
data.setdefault("X_train", X_train)
data.setdefault("y_train", y_train)
data.setdefault("X_val", X_val)
data.setdefault("y_val", y_val)


input_size = 32 * 32 * 3
num_classes = 10

# hidden_dim = []
# learning_rate = []
# epochs = []
# reg = []
# lr_decay = []

hidden_size = 64
learning_rate = 1e-3
epochs = 20
reg = 10
lr_decay = 0.98

""" 初始化两层神经网络模型 """
model = Layer(input_size, hidden_size, num_classes, reg=reg)

""" 初始化解决器 """
solver = Solver(model, data,
                optim_config={"learning_rate": learning_rate},
                num_epochs=epochs, batch_size=1000, print_every=48, lr_decay=lr_decay)

""" 训练模型 """
solver.train()

""" 将loss以及accuracy还有权重矩阵可视化 """
plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.plot(solver.loss_history, 'o')
plt.xlabel('Iteration')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()

show_net_weights(model)
# 笔记1:
# 当可视化神经网络第一层的权重矩阵时，重要的是要考虑神经网络的目的和它所处理的输入数据的类型。

# 通常，权重矩阵表示输入数据的学习特征。矩阵的每一行对应第一层的一个特定神经元，每一列对应输入数据中的一个特征或像素。
# 矩阵中的值表示每个神经元和每个特征之间的连接强度。

# 当权重矩阵可视化时，每一行都可以表示为一个图像，该图像显示输入数据中的哪些特征或像素被该神经元赋予了更大的权重。
# 这可以让我们了解神经网络正在学习什么，以及它是如何表示输入数据的。

print(solver.check_accuracy(X_test, y_test))



