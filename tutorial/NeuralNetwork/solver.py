import pickle

import numpy as np

from tutorial.NeuralNetwork import optim


class Solver(object):
    def __init__(self, model, data, **kwargs):
        """
        函数功能: 初始化一个Solver解决器，一个解决器里面有: 神经网络模型，训练数据，训练方法等...

        必要参数:
            - model: 一个具体的神经网络对象
            - data: 存储在字典中的数据集:
              'X_train': shape为(N_train, d_1, ..., d_k)的训练集；
              'X_val': shape为(N_val, d_1, ..., d_k)的验证集；
              'y_train': shape为(N_train,)的训练集标签；
              'y_val': shape为(N_val,)的验证集标签

        可选参数:
            - update_rule: optim.py里面的一个字段，用来指定更新规则。默认是'sgd'
            - optim_config: 包含将传递给所选更新规则的超参数的字典。每个更新规则需要不同的超参数(见optimize .py)，
              但所有更新规则都需要一个'learning_rate'参数，因此应该始终存在
            - lr_decay: 步长衰减的标量;在每次迭代之后，步长会乘以这个值
            - batch_size: 随机梯度下降中，每次用来训练的小批量的大小
            - num_epochs: 时代数(solver的迭代次数)
            - print_every: 整数; 每迭代print_every次之后就打印一次损失值
            - verbose: 布尔值; 如果设置为false，则在训练期间不会打印输出
            - num_train_samples: 训练样本的数量；默认值是1000；设置为None则用整个训练集
            - num_val_samples: 验证样本的数量；默认值为None，使用整个验证集
            - checkpoint_name: 如果不是None，那么在每次迭代时保存模型检查点

        其他自动初始化的参数:
            - epoch: 记录当前的时代
            - best_val_acc: 记录最高的验证集正确率
            - best_params: 记录模型里面的参数(W1,W2,b1,b2等权重矩阵信息)
            - loss_history: 记录训练过程中的loss
            - train_acc_history: 记录训练过程中的训练集的正确率
            - val_acc_history: 记录训练过程中的验证集的正确率
            - optim_configs: 配置文件,里面有更新梯度矩阵的方法,步长等等
        """

        # 笔记1:
        # **kwargs是Python中的一个特殊语法,用于接收传递给函数的可变数量的关键字参数(Keyword Arguments).
        # 它允许函数接受任意数量的关键字参数,并将它们作为字典(dictionary)的形式传递给函数内部
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        self.update_rule = kwargs.pop("update_rule", "sgd")
        # 笔记2:
        # 使用 kwargs.pop() 方法获取了关键字参数update_rule的值,并将其从字典中删除.如果字典中不存在这些键,则使用默认值.
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)
        self.num_train_samples = kwargs.pop("num_train_samples", 1000)
        self.num_val_samples = kwargs.pop("num_val_samples", None)
        self.checkpoint_name = kwargs.pop("checkpoint_name", None)

        """ 检查是否还有多余的参数 """
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        """ 其他参数 """
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.optim_configs = {}

        """ 检查optim.py文件中是否包含所需要的值 """
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        """ 为模型中的每个参数(权重矩阵,偏执矩阵)都设置一个配置文件,这样就可以单独为每个矩阵设置步长等参数 """
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            # 笔记1:
            # d 是self.optim_config的副本。self.optim_config是一个包含优化器的优化配置（例如学习率、动量）的字典。
            self.optim_configs[p] = d
            # 笔记2:
            # 为模型中的每个参数提供单独的优化配置。它确保每个参数都有自己的优化参数集，允许在优化过程中针对参数进行特定的定制。

    def _step(self):
        """
        函数功能: 更新梯度矩阵，在训练模型的时候被调用，不要手动调用
        """

        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size, replace=False)

        """ 计算损失值和梯度矩阵 """
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        """ 更新模型里面的权重矩阵以及对应的参数文件 """
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def _save_checkpoint(self):
        """
        函数功能: 保存并打印检查点
        """

        if self.checkpoint_name is None:
            return
        checkpoint = {
            "model": self.model,
            "update_rule": self.update_rule,
            "lr_decay": self.lr_decay,
            "optim_config": self.optim_config,
            "batch_size": self.batch_size,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "epoch": self.epoch,
            "loss_history": self.loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
        }
        filename = "%s_epoch_%d.pkl" % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)

    def check_accuracy(self, X, y, num_samples=None):
        """
        函数功能: 获得模型的准确率

        input:
            - X: 输入数据(N,d1, d2, ..., dn)
            - y: 隐藏层的特征维度
            - num_samples: 如果不是None，则对数据进行子采样，并且只在num_samples数据点上测试模型
            - batch_size: 一批数据的数据量
        return:
            - 准确率
        """
        N = X.shape[0]
        batch_size = self.batch_size
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for it in range(num_batches):
            start = it * batch_size
            end = (it + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        # 笔记1:
        # y_pred = np.hstack(y_pred)用于连接所有批的预测值，从而产生单个numpy数组，其中每列对应于给定样本的预测值。
        acc = np.mean(y_pred == y)
        return acc

    def train(self):
        """
        函数功能: 训练模型
        """

        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch  # 所花费的总的迭代次数

        for it in range(num_iterations):
            self._step()

            if self.verbose and it % self.print_every == 0 and it > 0:
                print("(Iteration %d / %d) loss: %f" % (it + 1, num_iterations, self.loss_history[-1]))
                # 笔记1:
                # loss_history[-1]返回的是最新的loss值，在这里也可以用loss_history[i]

            end_epoch = (it + 1) % iterations_per_epoch == 0
            # 笔记2:
            # end_epoch 是一个布尔值
            if end_epoch:
                self.epoch += 1
                for k in self.optim_configs:
                    """ 每经历一个"时代"就衰减一次步长 """
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay

            """ 检查训练集和验证集的准确率 """
            first_it = it == 0
            end_it = it == num_iterations - 1
            if first_it or end_it or end_epoch:
                train_acc = self.check_accuracy(self.X_train, self.y_train, num_samples=self.num_train_samples)
                val_acc = self.check_accuracy(self.X_val, self.y_val, num_samples=self.num_val_samples)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()
                if self.verbose:
                    print(
                        "(Epoch %d / %d) train acc: %f; val_acc: %f"
                        % (self.epoch, self.num_epochs, train_acc, val_acc)
                    )

                """ 筛选最佳模型参数 """
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        self.model.params = self.best_params


