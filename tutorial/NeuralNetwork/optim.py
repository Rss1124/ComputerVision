
def sgd(w, dw, config=None):
    """
    函数功能: 更新权重矩阵

    input:
        - w: 权重矩阵
        - dw: 梯度矩阵
        - config: 包含超参数值(如学习率、动量等)的字典。如果更新规则需要在多次迭代中缓存值，那么config也将保存这些缓存值。
    return:
        - w: 新的权重矩阵
        - config: 包含学习率的字典
    """

    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    w -= config["learning_rate"] * dw

    return w, config
