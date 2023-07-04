""" ----------------------------------------------- conv层(卷积层) ----------------------------------------------- """
import numpy as np

from BaseUtils.conv_utils import sliding_window, im2col, col2im


def conv_forward_naive(images, filters, b, conv_param):
    """
    函数功能: 卷积层的前向传播

    input:
        - images: shape为(N, C, H, W)的输入数据
        - filters: shape为(F, C, HH, WW)的过滤器
        - b: shape为(F,)的偏置矩阵
        - conv_param: 一个字典(包含一些卷积操作用的参数):
            - 'stride': 步长
            - 'pad': 将用于零填充输入的像素数(如果进行卷积操作时,过滤器无法覆盖整个图像区域)
            # 笔记1:
              在填充过程中，“垫”零应该沿输入的高度和宽度轴对称地放置(即在两侧相等)。注意不要直接修改原始输入images。

    Returns a tuple of:
        - out: shape为(N, F, OH, OW)的输出数据，OH = 1+(H+2*pad-HH)/stride，OW = 1+(W+2*pad-WW)/stride
        - cache: 缓存数据，包含(images, f, b, conv_param)

    """
    out = None
    images_pad = None

    N = images.shape[0]  # image的数量
    F = filters.shape[0]  # filter的数量

    """ image的尺寸 """
    H = images.shape[2]  # 高
    W = images.shape[3]  # 宽

    """ filter的尺寸 """
    HH = filters.shape[2]  # 高
    WW = filters.shape[3]  # 宽

    step = conv_param["stride"]  # filter的步长
    pad_num = conv_param["pad"]  # image的边缘扩展

    """ 获取窗口的维度 """
    OH = (H - HH + 2 * pad_num) // step + 1
    OW = (W - WW + 2 * pad_num) // step + 1
    out_dim = (OH, OW)

    if pad_num != 0:
        images_pad = np.zeros((N, images.shape[1], images.shape[2] + 2 * pad_num, images.shape[3] + 2 * pad_num))
    else:
        images_pad = images

    """ 获取out """
    out = np.zeros((N, F, OH, OW))

    """ 0填充 """
    pad_width = ((0, 0), (0, 0), (pad_num, pad_num), (pad_num, pad_num))
    images_pad = np.pad(images, pad_width, constant_values=0)
    # 笔记1:
    # pad_width 内部的元组,就代表对应的维度
    # 在这里,image(N,C,H,W)的前两个维度不进行拓展"0",后面两个维度则会进行拓展,拓展的宽度为pad_num,拓展的值为constant_values

    for i in range(N):
        for j in range(F):
            """ 获取滤波器滑动后的窗口集 """
            x = sliding_window(images_pad[i], (filters.shape[1], HH, WW), conv_param["stride"])

            """ 将窗口集和滤波器处理为二维矩阵 """
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            x = x.reshape(x.shape[0], -1)
            w = filters[j]
            w = w.reshape(-1, )

            """ 计算点积 """
            convolution = (x.dot(w) + b[j]).reshape(out_dim)
            out[i][j] = convolution

    cache = (images_pad, filters, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    函数功能: 卷积层的反向传播

    input:
        - dout: shape为(N, F, OH, OW)的上一层导数，OH = 1+(H+2*pad-HH)/stride，OW = 1+(W+2*pad-WW)/stride
        - cache: 在前向传播中保存的缓存数据

    returns a tuple of:
        - dx: x(image)的梯度，shape为(N, C, H, W)
        - dw: w(filter)的梯度，shape为(F, C, HH, WW)
        - db: b的梯度，shape为(F,)
    """
    dx, dw, db = None, None, None

    x, w, b, conv_param = cache  # x==image, w==filter, b==b

    dx = np.zeros(shape=x.shape)
    dw = np.zeros(shape=w.shape)
    db = np.zeros(shape=b.shape)

    N = dout.shape[0]  # image的数量
    F = dout.shape[1]  # filter的数量
    OH = dout.shape[2]
    OW = dout.shape[3]
    step = conv_param["stride"]  # filter的步长
    pad = conv_param["pad"]  # 边缘的拓展pad

    for i in range(N):
        for j in range(F):
            for m in range(OH):
                for n in range(OW):
                    """ 计算dx """
                    dx[i, :, step * m:w.shape[2] + step * m, step * n:w.shape[3] + step * n] += w[j].dot(
                        dout[i][j][m][n])
                    """ 计算dw """
                    dw[j] += x[i, :, step * m:w.shape[2] + step * m, step * n:w.shape[3] + step * n].dot(
                        dout[i][j][m][n])
                    """ 计算db """
                    db[j] += dout[i][j][m][n]

    dx = dx[:,:,pad:x.shape[2]-pad,pad:x.shape[3]-pad]
    return dx, dw, db

def conv_forward_im2col(x, w, b, conv_param):
    """
    函数功能: 使用im2col算法来实现卷积层的前向传播

    input:
        - x: shape为(N,C,H,W)的图像集
        - w: shape为(F,C,HH,WW)的滤波器集
        - b: shape为(F, )偏执项
        - conv_param: 卷积参数,包含(stride, pad)

    returns a tuple of:
        - out: shape为(N, F, OH, OW)的输出数据，OH = 1+(H+2*pad-HH)/stride，OW = 1+(W+2*pad-WW)/stride
        - cache: 缓存数据，包含(x, w, b, conv_param, x_cols)
    """
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param["stride"], conv_param["pad"]

    """ 检查OH, OW是否正常 """
    assert (W + 2 * pad - filter_width) % stride == 0, "width does not work"
    assert (H + 2 * pad - filter_height) % stride == 0, "height does not work"

    """ 计算OH,OW """
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1

    """ 初始化out矩阵 """
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    """ 将x(rgb图像)映射到二维矩阵x_cols上, 方便后续计算 """
    x_cols = im2col(x, w.shape[2], w.shape[3], conv_param, out_height, out_width)

    """ 计算点积 """
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    """ 将数据存储在缓存中 """
    cache = (x, w, b, conv_param, x_cols)
    return out, cache

def conv_backward_col2im(dout, cache):
    """
    函数功能: 使用im2col算法来实现卷积层的反向传播

    input:
        - dout: shape为(N, F, OH, OW)的上一层导数，OH = 1+(H+2*pad-HH)/stride，OW = 1+(W+2*pad-WW)/stride
        - cache: 在前向传播中保存的缓存数据

    returns a tuple of:
        - dx: x(image)的梯度，shape为(N, C, H, W)
        - dw: w(filter)的梯度，shape为(F, C, HH, WW)
        - db: b的梯度，shape为(F,)
    """

    """ 从缓存中获取数据 """
    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param["stride"], conv_param["pad"]

    """ 计算db """
    db = np.sum(dout, axis=(0, 2, 3))
    # 笔记1:
    # np.sum(dout, axis=(0, 2, 3)) 表示对dout的第0,第2和第3个轴进行求和，得到一个降维后的一维数组 db
    # 在这里是对所有图片的每个通道进行的求和

    num_filters, _, filter_height, filter_width = w.shape
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)

    """ 计算dw """
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    """ 计算dx """
    dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)

    """ 将dx_cols映射到4维矩阵上 """
    x_padded, dx = col2im(
        dx_cols,
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.shape[3],
        filter_height,
        filter_width,
        pad,
        stride,
    )

    return dx, dw, db

