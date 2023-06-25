from numpy.lib.stride_tricks import as_strided

def sliding_window(image, window_size, step):
    """
    函数功能:滤波器在image上根据步长值进行滑动，获取所有窗口

    input:
        - image: 一张RGB图像,shape为(C,H,W)
        - window_size: 滤波器的尺寸(C,HH,WW)

    return:
        - window_view: shape为(OH, OW, C, HH, WW)的输出数据
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
