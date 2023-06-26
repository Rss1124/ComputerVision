""" ---------------------------------------- max_pool层(最大池化层) ---------------------------------------- """
import numpy as np

from BaseUtils.conv_utils import sliding_window


def max_pool_forward_naive(x, pool_param):
    """
    函数功能: 最大池化层的前向传播

    Inputs:
        - x: shape为(N, C, H, W)的池化层数据
        - pool_param: 字典（池化层参数）:
            - 'pool_height': 池化窗口的高度
            - 'pool_width': 池化窗口的宽度
            - 'stride': 池化窗口的步长
            # 笔记1:
              最大池化层可以不用填充
            # 笔记2:
              通常情况下，最大池化层没有重叠窗口

    Returns a tuple of:
        - out: shape为(N,C,OH,OW)的输出数据，OH = 1+(H-pool_height)/stride，OW = 1+(W-pool_width)/stride
        - cache: 缓存数据，包含(images, pool_param)
    """
    out = None

    N = x.shape[0]  # 卷积层的个数（图像的个数）
    C = x.shape[1]  # 卷积层的通道数
    H = x.shape[2]  # 卷积层的高度
    W = x.shape[3]  # 卷积层的宽度
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    step = pool_param['stride']

    """ 计算out的尺寸 """
    OH = (H - HH) // step + 1
    OW = (W - WW) // step + 1

    """ 初始化out矩阵 """
    out = np.zeros((N, C, OH, OW))

    """ 滑动窗口做最大池化操作 """
    for i in range(N):
        for j in range(C):
            windows = sliding_window(image=x[i][j], window_size=(HH, WW), step=step)
            windows = windows.reshape(-1, windows.shape[2], windows.shape[3])
            windows = windows.reshape(windows.shape[0], -1)
            result = np.max(windows, axis=1)
            out[i][j] = result.reshape(OH, OW)

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    函数功能: 最大池化层的反向传播

    Inputs:
    - dout: shape为(N,C,OH,OW)的上一层导数
    - cache: 在前向传播中保存的缓存数据(x, pool_param)

    Returns:
    - dx: x的梯度，shape为((N,C,H,W))
    """

    x, pool_param = cache
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    step = pool_param['stride']
    N, C, OH, OW = dout.shape
    dx = np.zeros(x.shape)
    for i in range(N):
        for j in range(C):
            for m in range(OH):
                for n in range(OW):
                    index = np.argmax(x[i][j][m * step: HH + m * step, n * step: WW + n * step])
                    index_2d = np.unravel_index(index, (HH, WW))
                    temp = np.zeros((HH, WW))
                    temp[index_2d] = dout[i][j][m][n]
                    dx[i][j][m * step: HH + m * step, n * step: WW + n * step] = temp

    return dx
