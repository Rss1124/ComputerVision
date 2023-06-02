from matplotlib import pyplot as plt

from BaseUtils.data_utils import get_CIFAR10_data
from BaseUtils.features_utils import *
from tutorial.ImageClassifier.SVM.Linear_Classifier import Classifier
from tutorial.ImageClassifier.SVM.softmax import softmax_loss_vectorized

X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
X_train = X_train.transpose(0,2,3,1)
X_val = X_val.transpose(0,2,3,1)
X_test = X_test.transpose(0,2,3,1)
X_dev = X_dev.transpose(0,2,3,1)
num_color_bins = 10
""" 初始化"特征函数"数组 """
# feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
feature_fns = [hog_feature, color_histogram_hsv]
""" 对数据集使用不同特征函数进行特征处理 """
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])

learning_rates = [1e-9, 1e-8, 1e-7]
regularization_strengths = [5e4, 5e5, 5e6]

results = {}
best_val = -1
best_svm = None

for i in learning_rates:
    for j in regularization_strengths:
        params = (i, j)
        svm_classifier = Classifier()
        lr = i
        reg = j
        svm_classifier.train(function="softmax", X=X_train_feats, y=y_train, learning_rate=lr, reg=reg, num_iters=400, batch_size=1000, verbose=True)
        y_train_pred = svm_classifier.predict(X_train_feats)
        y_train_accuracy = np.mean(y_train == y_train_pred)
        y_val_pred = svm_classifier.predict(X_val_feats)
        y_val_accuracy = np.mean(y_val == y_val_pred)
        accuracy = (y_train_accuracy, y_val_accuracy)

        if y_val_accuracy > best_val:
            best_val = y_val_accuracy
            best_svm = svm_classifier
        results[params] = accuracy

for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
        lr, reg, train_accuracy, val_accuracy))
print('best validation accuracy : %f' % best_val)

y_test_pred = best_svm.predict(X_test_feats)
test_accuracy = np.mean(y_test == y_test_pred)
print(test_accuracy)

examples_per_class = 8
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for cls, cls_name in enumerate(classes):
    idxs = np.where((y_test != cls) & (y_test_pred == cls))[0]
    idxs = np.random.choice(idxs, examples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
        plt.imshow(X_test[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls_name)
plt.show()
