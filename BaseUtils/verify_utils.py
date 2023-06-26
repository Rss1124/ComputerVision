import numpy as np

from BaseUtils.gradient_check import eval_numerical_gradient_array
from tutorial.NeuralNetwork.layer_utils.conv import conv_forward, conv_backward
from tutorial.NeuralNetwork.layer_utils.max_pool import max_pool_forward_naive, max_pool_backward_naive


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def verify_conv_forward():
    """
    函数功能: 验证卷积层的前向传播方法
    """
    x_shape = (2, 3, 4, 4)
    w_shape = (3, 3, 4, 4)
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
    w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
    b = np.linspace(-0.1, 0.2, num=3)

    conv_param = {'stride': 2, 'pad': 1}
    out, _ = conv_forward(x, w, b, conv_param)
    correct_out = np.array([[[[-0.08759809, -0.10987781],
                              [-0.18387192, -0.2109216]],
                             [[0.21027089, 0.21661097],
                              [0.22847626, 0.23004637]],
                             [[0.50813986, 0.54309974],
                              [0.64082444, 0.67101435]]],
                            [[[-0.98053589, -1.03143541],
                              [-1.19128892, -1.24695841]],
                             [[0.69108355, 0.66880383],
                              [0.59480972, 0.56776003]],
                             [[2.36270298, 2.36904306],
                              [2.38090835, 2.38247847]]]])

    # Compare your output to ours; difference should be around e-8
    print('Testing conv_forward_naive')
    print('difference: ', rel_error(out, correct_out))

def verify_conv_backward():
    """
    函数功能: 验证卷积层的反向传播算法
    """
    np.random.seed(231)
    x = np.random.randn(4, 3, 5, 5)
    w = np.random.randn(2, 3, 3, 3)
    b = np.random.randn(2, )
    dout = np.random.randn(4, 2, 5, 5)
    conv_param = {'stride': 1, 'pad': 1}

    dx_num = eval_numerical_gradient_array(lambda x: conv_forward(x, w, b, conv_param)[0], x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: conv_forward(x, w, b, conv_param)[0], w, dout)
    db_num = eval_numerical_gradient_array(lambda b: conv_forward(x, w, b, conv_param)[0], b, dout)

    out, cache = conv_forward(x, w, b, conv_param)
    dx, dw, db = conv_backward(dout, cache)

    # Your errors should be around e-8 or less.
    print('Testing conv_backward_naive function')
    print('dx error: ', rel_error(dx, dx_num))
    print('dw error: ', rel_error(dw, dw_num))
    print('db error: ', rel_error(db, db_num))

def verify_maxPool_forward():
    """
    函数功能: 验证池化层的前向传播算法
    """
    x_shape = (2, 3, 4, 4)
    x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
    pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}

    out, _ = max_pool_forward_naive(x, pool_param)

    correct_out = np.array([[[[-0.26315789, -0.24842105],
                              [-0.20421053, -0.18947368]],
                             [[-0.14526316, -0.13052632],
                              [-0.08631579, -0.07157895]],
                             [[-0.02736842, -0.01263158],
                              [0.03157895, 0.04631579]]],
                            [[[0.09052632, 0.10526316],
                              [0.14947368, 0.16421053]],
                             [[0.20842105, 0.22315789],
                              [0.26736842, 0.28210526]],
                             [[0.32631579, 0.34105263],
                              [0.38526316, 0.4]]]])

    # Compare your output with ours. Difference should be on the order of e-8.
    print('Testing max_pool_forward_naive function:')
    print('difference: ', rel_error(out, correct_out))

def verify_maxPool_backward():
    """
    函数功能: 验证池化层的反向传播算法
    """
    np.random.seed(231)
    x = np.random.randn(3, 2, 8, 8)
    dout = np.random.randn(3, 2, 4, 4)
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    dx_num = eval_numerical_gradient_array(lambda x: max_pool_forward_naive(x, pool_param)[0], x, dout)

    out, cache = max_pool_forward_naive(x, pool_param)
    dx = max_pool_backward_naive(dout, cache)

    # Your error should be on the order of e-12
    print('Testing max_pool_backward_naive function:')
    print('dx error: ', rel_error(dx, dx_num))