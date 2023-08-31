import numpy as np
from PIL import Image
from imageio import imread
from matplotlib import pyplot as plt

from BaseUtils.data_utils import get_CIFAR10_data
from BaseUtils.verify_utils import verify_conv_forward_naive, verify_conv_backward_naive
from BaseUtils.vis_utils import visualize_grid
from tutorial.NeuralNetwork.cnn import ThreeLayerConvNet
from tutorial.NeuralNetwork.layer_utils.conv import conv_forward_naive, conv_backward_naive, conv_forward_im2col
from tutorial.NeuralNetwork.solver import Solver

# X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
#
# kitten = imread('C:/Users/60434/Desktop/study/assignment2_colab/assignment2/cs231n/notebook_images/kitten.jpg')
# puppy = imread('C:/Users/60434/Desktop/study/assignment2_colab/assignment2/cs231n/notebook_images/puppy.jpg')
# # kitten is wide, and puppy is already square
# d = kitten.shape[1] - kitten.shape[0]
# kitten_cropped = kitten[:, d // 2:-d // 2, :]
#
# img_size = 200  # Make this smaller if it runs too slow
# resized_puppy = np.array(Image.fromarray(puppy).resize((img_size, img_size)))
# resized_kitten = np.array(Image.fromarray(kitten_cropped).resize((img_size, img_size)))
# x = np.zeros((2, 3, img_size, img_size))
# x[0, :, :, :] = resized_puppy.transpose((2, 0, 1))
# x[1, :, :, :] = resized_kitten.transpose((2, 0, 1))
#
# # Set up a convolutional weights holding 2 filters, each 3x3
# w = np.zeros((2, 3, 3, 3))
#
# # The first filter converts the image to grayscale.
# # Set up the red, green, and blue channels of the filter.
# w[0, 0, :, :] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
# w[0, 1, :, :] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
# w[0, 2, :, :] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
#
# # Second filter detects horizontal edges in the blue channel.
# w[1, 0, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
#
# # Vector of biases. We don't need any bias for the grayscale
# # filter, but for the edge detection filter we want to add 128
# # to each output so that nothing is negative.
# b = np.array([0, 128])
#
# # Compute the result of convolving each input in x with each filter in w,
# # offsetting by b, and storing the results in out.
# out, _ = conv_forward_im2col(x, w, b, {'stride': 1, 'pad': 0})
#
#
# def imshow_no_ax(img, normalize=True):
#     """ Tiny helper to show images as uint8 and remove axis labels """
#     if normalize:
#         img_max, img_min = np.max(img), np.min(img)
#         img = 255.0 * (img - img_min) / (img_max - img_min)
#     plt.imshow(img.astype('uint8'))
#     plt.gca().axis('off')
#
#
# # Show the original images and the results of the conv operation
# plt.subplot(2, 3, 1)
# imshow_no_ax(puppy, normalize=False)
# plt.title('Original image')
# plt.subplot(2, 3, 2)
# imshow_no_ax(out[0, 0])
# plt.title('Grayscale')
# plt.subplot(2, 3, 3)
# imshow_no_ax(out[0, 1])
# plt.title('Edges')
# plt.subplot(2, 3, 4)
# imshow_no_ax(kitten_cropped, normalize=False)
# plt.subplot(2, 3, 5)
# imshow_no_ax(out[1, 0])
# plt.subplot(2, 3, 6)
# imshow_no_ax(out[1, 1])
# plt.show()
#
# """ 验证传播算法的正确性 """
# verify_conv_forward_naive()
# verify_conv_backward_naive()


np.random.seed(231)

X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
data = {}
data.setdefault("X_train", X_train)
data.setdefault("y_train", y_train)
data.setdefault("X_val", X_val)
data.setdefault("y_val", y_val)

num_train = 100
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=200, reg=0.001)

solver = Solver(
    model,
    small_data,
    num_epochs=1,
    batch_size=50,
    update_rule='adam',
    optim_config={'learning_rate': 1e-3,},
    verbose=True,
    print_every=20
)
solver.train()

print(
    "Small data training accuracy:",
    solver.check_accuracy(small_data['X_train'], small_data['y_train'])
)

print(
    "Small data validation accuracy:",
    solver.check_accuracy(small_data['X_val'], small_data['y_val'])
)

plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

grid = visualize_grid(model.params['W1'].transpose(0, 2, 3, 1))
plt.imshow(grid.astype('uint8'))
plt.axis('off')
plt.gcf().set_size_inches(5, 5)
plt.show()
