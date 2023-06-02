import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter


def extract_features(imgs, feature_fns, verbose=False):
    """
    函数功能: 提取图像的特征，给定图像的像素数据和可以在单个图像上操作的几个特征函数，将所有特征函数应用于所有图像，
    连接每个图像的特征向量并将所有图像的特征存储在单个矩阵中。

    input:
        - imgs : 图像集，shape为(N,H,W,C) .
        - feature_fns : k个特征函数的列表。第i个特征函数应该以一个(H,W,D)的数组作为输入，并返回一个长度为F_i的(一维)数组。
        - verbose : 布尔值，如果为true，则打印运行过程

    return:
        - imgs_features: 所有图像的特征，shape为(N, F_1 + ...  + F_k)，每一列代表图像在某一个特征函数上的特征值

    """

    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    """ 默认使用第一个图像来决定整个图像集的特征维度,并提取第一个图像的特征 """
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        """ 使用所有的特征函数来计算第一个图像 """
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, "Feature functions must be one-dimensional"
        """ 获取图像在不同特征函数下的特征维度 """
        feature_dims.append(feats.size)
        """ 将第一个图像的所有特征值保存下来 """
        first_image_features.append(feats)

    """ 获取所有的特征的维度个数(F_1 + ...  + F_k) """
    total_feature_dim = sum(feature_dims)
    """ 初始化特征数组 """
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T

    """ 提取其余图像的特征 """
    for i in range(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            # 笔记1:
            # 使用zip函数能够在for循环中,同时操作多个值
            next_idx = idx + feature_dim
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i % 1000 == 999:
            print("Done extracting features for %d / %d images" % (i + 1, num_images))

    return imgs_features


def rgb2gray(rgb):
    """
    函数功能: 将RGB图像转换为灰度图像

    input:
        - rgb: RGB图像

    return:
        - gray: 灰度图像

    """

    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])
    # 笔记1:
    # [..., :3] 是 NumPy 的切片操作，用于选择数组的最后一个维度上的前三个元素，即选择RGB通道的数据。放在二维空间上就是选择前三列数据
    # 这个切片操作可以确保只选择 RGB 通道的数据进行后续计算。
    # [0.299, 0.587, 0.144] 是一个长度为3的一维数组，表示 RGB 通道的权重。
    # 根据灰度转换公式，这些权重对应于每个通道的相对贡献度。


def hog_feature(im):
    """
    函数功能: 获取图像的梯度直方图

    input:
        - im : 一个灰度图像或者是rgb图像

    return:
        - feat : 图像的梯度直方图

    Reference:
        - Histograms of Oriented Gradients for Human Detection
        - Navneet Dalal and Bill Triggs, CVPR 2005

    """

    """ 如果输入的图像是rgb图像,将其转化为2维的灰度图像 """
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.at_least_2d(im)

    sx, sy = image.shape  # 图像大小
    orientations = 9
    # 笔记1:
    # orientation(梯度箱数)
    # 梯度是指图像中像素值变化最剧烈的方向。梯度箱数也就是梯度方向的数量。
    # 在这种情况下，使用9个梯度方向范围（或9个角度范围）将梯度方向离散化。因此，对于每个像素的梯度方向，它将被分配到9个分箱之一。
    # 通过将梯度方向离散化为多个分箱，可以将梯度信息转换为直方图形式，以便表示图像中的梯度分布情况，并用于后续的特征提取和图像分类任务。
    cx, cy = (8, 8)  # 细胞的大小
    # 笔记2:
    # 在计算机视觉和图像处理中，"细胞"（cell）通常指的是图像的一个局部区域或分块。
    # 这些局部区域通常是规则的矩形或正方形，用于对图像进行分割和处理，每个细胞通常会提取一组特征，例如颜色直方图、梯度方向直方图等，
    # 以描述该细胞内部的局部图像信息
    n_cellsx = int(np.floor(sx / cx))  # 在x轴上的细胞个数
    n_cellsy = int(np.floor(sy / cy))  # 在y轴上的细胞个数

    """ 初始化x轴,y轴方向上的的梯度(差分) """
    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)

    """ 计算每个像素在x轴,y轴方向上的梯度 """
    gx[:, :-1] = np.diff(image, n=1, axis=1)
    gy[:-1, :] = np.diff(image, n=1, axis=0)
    # 笔记3:
    # 在这里，计算x轴方向上的梯度的方法是计算image数组在第一个轴上(即行方向)相邻元素的差值
    # 同理，计算y轴方向上的梯度是计算image数组在列方向上相邻元素的差值
    # 笔记4:
    # gx,gy 的形状与 image 相同，但是由于计算方法的限制，会导致最后一行和最后一列的差分无法计算，所以差分操作会导致少一个元素

    """ 计算每个像素”真正的“梯度(矢量) """
    # 笔记5:
    # 在像素中，梯度表示像素值变化，所以梯度是一个矢量，有方向和大小
    # 计算梯度的大小使用L2距离公式
    # 计算梯度的方向需要用到反正切函数，arctan * (180/pi) = 角度值
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # 梯度值
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # 梯度方向(角度)
    # 笔记6:
    # 在角度值后面+90，是为了调整角度范围，将最终的角度范围调整在[0, 180]

    """ 初始化方向直方图 """
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))

    """ 将一个图像的所有特征分为9种情况(梯度箱数的值) """
    for i in range(orientations):
        """ 将角度(180°)切割为9份,每份代表一个梯度方向区间 """
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1), grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i, temp_ori, 0)
        # 笔记7:
        # 每次循环会将grad_ori区间内[180 / orientations * i , 180 / orientations * (i + 1)]的值保留,其他区间的值设为0
        # 这两步操作可以将梯度方向 grad_ori 分配到离散的方向区间中，使得每个梯度方向只在对应的区间内有非零值
        # 通过这种方式，可以对梯度方向进行量化和编码，以便后续的特征提取和分析
        """ 获取对应梯度方向区间里面的梯度值 """
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        """ 对梯度值做平滑处理,以此来减轻噪声的影响 """
        orientation_histogram_temp = uniform_filter(temp_mag, size=(cx, cy))
        # 笔记8:
        # uniform_filter 是一种平滑滤波器，用于对输入数组进行平滑处理。
        # 该函数的结果是一个与输入数组大小相同的数组，其中的每个元素表示对应位置周围窗口内数值的均值。
        # 如果过滤器处在图像的边缘，会通过几种边界处理模式来处理，默认使用reflect模式
        """ 将处在[round(cx / 2):: cx, round(cy / 2):: cy]位置的元素提取出来,用于最终的梯度直方图 """
        orientation_histogram[:, :, i] = orientation_histogram_temp[
            round(cx / 2) :: cx, round(cy / 2) :: cy
        ].T
        # 笔记9:
        # 当使用 "reflect" 模式时，窗口在边界处通过反射地复制邻近的像素来进行扩展。
        # 这意味着窗口内的像素值是通过将窗口映射到输入数组的边界处，并利用边界像素的反射镜像来计算的。
        # 样例: 如果将窗口应用在数组[0][0]的位置上，那么首先会把该元素放在中心位置，然后直接复制周围元素，来填补边缘的空白
        # [[1, 2, 3],         [[1, 1, 2],
        #  [4, 5, 6],   ==>    [1, 1, 2],  ==> 所以[0][0]位置的值应该是(1+1+2+1+1+2+4+4+5)/9 = 2
        #  [7, 8, 9]]          [4, 4, 5]]


    # 笔记10:
    # ravel() 是将多维数组 orientation_histogram 平铺（展开）为一维数组的操作

    # plt.figure(figsize=(10, 8))
    # for i in range(orientation_histogram.shape[2]):
    #     plt.subplot(3, 3, i + 1)  # 根据直方图通道数量调整子图位置
    #     plt.imshow(orientation_histogram[:, :, i])
    #     plt.colorbar()  # 添加颜色条
    #
    #     # 在每个子图上添加文本标签显示每个元素的值
    #     for (x, y), value in np.ndenumerate(orientation_histogram[:, :, i]):
    #         plt.text(y, x, f'{value:.2f}', color='w', ha='center', va='center')
    #
    # plt.suptitle('Orientation Histograms')
    # plt.tight_layout()
    # plt.show()

    return orientation_histogram.ravel()

def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    函数功能: 使用色相计算图像的颜色直方图

    input:
        - im: RGB图像,shape为(H,W,C)
        - nbin: 直方图的箱数 (默认值: 10)
        - xmin: 最小的像素值 (默认值: 0)
        - xmax: 最大的像素值 (默认值: 255)
        - normalized: 是否做归一化 (默认值: True)

    return:
        - imhist: 长度为nbin的一数组，表示输入图像的色调上的颜色直方图
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    hsv = matplotlib.colors.rgb_to_hsv(im / xmax) * xmax
    # 笔记1:
    # hsv是HSV颜色空间。HSV表示色彩的三个属性：色调（Hue）、饱和度（Saturation）和明度（Value）。
    # HSV颜色空间相对于RGB颜色空间更符合人类对颜色的感知方式。
    """ 计算图像的色相通道的直方图 """
    imhist, bin_edges = np.histogram(hsv[:, :, 0], bins=bins, density=normalized)
    imhist = imhist * np.diff(bin_edges)
    return imhist
