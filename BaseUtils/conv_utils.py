import numpy as np
from numpy.lib.stride_tricks import as_strided

def sliding_window(image, window_size, step):
    """
    函数功能:滤波器在image上根据步长值进行滑动，获取所有窗口

    input:
        - image: 一张RGB图像,shape为(C,H,W)或者(H,W)
        - window_size: 滤波器的尺寸(C,HH,WW)或者(HH,WW)
        - step: 滤波器的步长

    return:
        - window_view:
            - 如果image为二维，则返回shape为(OH, OW, HH, WW)的输出数据
            - 如果image为三维，则返回shape为(OH, OW, C, HH, WW)的输出数据
    """

    """ 计算窗口在图像上滑动的步幅 """
    stride = image.strides
    # 笔记1:
    # strides函数返回的是一个元组，里面的数据量与image的维数相同
    # 具体的数值代表在内存中，image矩阵中的某个元素在某个维度上移动一个位置所需要的字节数
    # 举例:
    # image = np.random.randn(4, 4)，image矩阵内的元素是浮点数，所以矩阵内的每个元素都是8byte
    # 在这个4*4的矩阵中，一个元素如果要垂直移动一个位置，由于矩阵在内存中默认是按行序存储的，所以它需要移动4个元素位置，才能到达目标位置
    # 这就需要4*8=32个byte。同理，如果元素要水平移动一个位置，只需要移动一次，那就需要1*8=8byte
    # image.strides = (32, 8)

    """ 计算卷积之后的shape以及滤波器的stride """
    if image.ndim == 3:
        channel, height, width = image.shape
        window_channel, window_height, window_width = window_size
        output_shape = (
                        (height - window_height) // step + 1,
                        (width - window_width) // step + 1,
                        window_channel,
                        window_height,
                        window_width)
        strides = (stride[1] * step, stride[2] * step) + stride

    elif image.ndim == 2:
        height, width = image.shape
        window_height, window_width = window_size
        output_shape = ((height - window_height) // step + 1,
                        (width - window_width) // step + 1,
                        window_height,
                        window_width)
        strides = (stride[0] * step, stride[1] * step) + stride

    """ 获取通过滤波器滑动得到的 ( height - window_height + 1 ) * ( width - window_width + 1 )个窗口 """
    window_view = as_strided(image, shape=output_shape, strides=strides)
    # 笔记2:
    # as_strided()函数需要注意shape与strides的len相同（即两个元组的长度要相同）
    # 举例:
    # 当image.nidm为3时，stride的长度为3，那么strides的长度就变为了6，其实就需要output_shape的长度也为6

    # 笔记3: 如何理解as_strided()函数中的strides参数？（要理解strides参数需要结合shape参数来一起理解）
    # 举例:
    # input = np.random.randn(3,4,4)
    # window_size = (3,2,2)
    # step = 2
    # 我们可以计算出out_shape为(2,2,3,2,2)，以及strides为(64,16,128,32,8)
    # 这个时候就需要拆解每个数值所代表的含义了，第一个2代表"滤波器"（window）在"图像"（image）中沿着纵坐标滑动的次数。
    # 同理，第二个2代表沿着横坐标滑动的次数。所以strides中的前两个数值(64,16)分别代表着window沿着步长移动到下一个位置所需要的byte，
    # 因为window不需要在深度上进行移动，所以只需要两个数值。而strides中的后面三个数值(128,32,8)则代表着window内部元素的移动情况。
    # 128代表着window内部元素沿着z轴移动所需要的byte，32代表着window内部元素沿着y轴移动所需要的byte，8代表着window内部元素沿着x轴
    # 移动所需要的byte,
    return window_view

def im2col(images, window_height, window_width, conv_param, OH, OW):
    """
    函数功能:将shape为(N,C,H,W)的图像映射到二维矩阵cols上面去

    input:
        - image: 一张RGB图像,shape为(N,C,H,W)
        - window_height: 滤波器的高
        - window_width: 滤波器的宽
        - conv-param: 卷积的参数, 包含(stride ,pad)
        - OH: 卷积之后的输出窗口的高
        - OW: 卷积之后的输出窗口的宽

    return:
        - cols: shape为(C * window_height * window_width, N * OH * OW)
    """

    N, C, H, W = images.shape
    step = conv_param["stride"]
    pad_num = conv_param["pad"]
    cols = np.zeros((C * window_height * window_width, N * OH * OW))

    if pad_num != 0:
        images_pad = np.zeros((N, images.shape[1], images.shape[2] + 2 * pad_num, images.shape[3] + 2 * pad_num))
    else:
        images_pad = images

    pad_width = ((0, 0), (0, 0), (pad_num, pad_num), (pad_num, pad_num))
    images_pad = np.pad(images, pad_width, constant_values=0)

    for c in range(C):
        for oh in range(OH):
            for ow in range(OW):
                for w_h in range(window_height):
                    for w_w in range(window_width):
                        row = c * window_height * window_width + w_h * window_height + w_w
                        # 笔记1:
                        # c * field_h * field_w 代表着在不同通道上的"初始窗口"在一维数组上的的映射地址
                        # f_h * field_h + f_w 代表着每个窗口内部的地址
                        for n in range(N):
                            column = oh * OW * N + ow * N + n
                            # 笔记2:
                            # hh * WW * N 代表着窗口在所有图像中，垂直方向的偏移
                            # ww * N 代表着窗口在所有图像中，水平方向的偏移
                            # n 代表着窗口在不同图像中的偏移
                            cols[row, column] = images_pad[n, c, step * oh + w_h, step * ow + w_w]
                            # 笔记3:
                            # cols与image_padded的对应关系如下:
                            # 假设参数为: N=2, C=3, H=4, W=4, OH=2, OW=2, window_height=2, window_width=2
                            # 最终我们会得到一个shape为(12, 8)的cols矩阵,12代表一个区块所包含的元素的个数,8代表区块的个数
                            # 并且当column= 0,2,4,6 时代表着第一个图像的四个区块, 当column= 1,3,5,7 时代表着第二图像的四个区块

                            # 其中row可以解析到image_padded.shape[1],image_padded.shape[2],image_padded.shape[3]上
                            # column可以解析到image_padded.shape[0],image_padded.shape[2],image_padded.shape[3]上

                            # 举例:clos[5,4]
                            # 5代表着区块里的第六个元素,意味着在第二层的通道上
                            # 4代表着第一个图像的第三个区块
                            # 这样我们就将cols[5,4]映射到了第一个图像的第三个区块的第六个元素也就是image_padded[0,1,2,1]

    return cols

def col2im(cols, N, C, H, W, window_height, window_width, padding, stride):
    """
    函数功能:将二维矩阵cols映射到shape为(N,C,H,W)的四维矩阵上面去，在这里有两种映射的方法：
            一种是im2col的逆运算，另一种是将dx映射回去

    input:
        - cols: shape为(C * window_height * window_width, N * OH * OW)的二维矩阵,由image映射而来
        - N: image的数量
        - C: image的通道数
        - H: image的高
        - W: image的宽
        - window_height: 滤波器的高
        - window_width: 滤波器的宽
        - padding: 填充宽度
        - stride: 步长

    return:
        - image: shape为(N,C,H,W)的原图像
        - dx: 反向传播中, x的梯度
    """

    OH = (H + 2 * padding - window_height) // stride + 1
    OW = (W + 2 * padding - window_width) // stride + 1
    x_padded = np.zeros((N,C,H + 2 * padding, W + 2 * padding), dtype=cols.dtype)
    dx = np.zeros((N,C,H + 2 * padding, W + 2 * padding), dtype=cols.dtype)

    for c in range(C):
        for w_h in range(window_height):
            for w_w in range(window_width):
                row = c * window_height * window_width + w_h * window_height + w_w
                for oh in range(OH):
                    for ow in range(OW):
                        for n in range(N):
                            column = oh * OW * N + ow * N + n
                            """ im2col的逆运算 """
                            x_padded[n, c,  stride * oh + w_h, stride * ow + w_w] = cols[row, column]
                            # 笔记1:
                            # 将cols映射回去只需要将cols里面的元素回填到x_padded中即可
                            """ 将dx映射到四维矩阵中 """
                            dx[n, c,  stride * oh + w_h, stride * ow + w_w] += cols[row, column]
                            # 笔记2:
                            # 在卷积层反向传播中计算x的梯度dx,需要叠加元素,所以这里的映射操作用了操作符"+="

    if padding > 0:
        image = x_padded[:, :, padding:-padding, padding:-padding]
        dx = dx[:, :, padding:-padding, padding:-padding]
    else:
        image = x_padded
    return image, dx
