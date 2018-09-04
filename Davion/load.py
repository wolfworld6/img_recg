import cv2
import numpy as np
import os
from skimage import feature as ft

from sklearn.model_selection import train_test_split
import pickle

from Davion import lof


def load(pklf):
    """
    通过载入数据文件返回训练数据和测试数据
    :param pklf: 存有特征和标签的pkl文件
    :return: 训练数据和测试数据
    """
    fr = open(pklf, 'rb')
    inf = pickle.load(fr)
    X = []
    y = []
    for x in inf:
        X.append(x[:9])
        # print(x[:9])
        y.append(x[9])
    X = np.array(X)
    y = np.array(y)
    X_norm = []
    X_abnorm = []
    y_norm = []
    y_abnorm = []
    for i in range(len(y)):
        if y[i] == 1:
            X_norm.append(X[i])
            y_norm.append(y[i])
        else:
            X_abnorm.append(X[i])
            y_abnorm.append(y[i])
    X_norm_train, X_norm_test, y_norm_train, y_norm_test = train_test_split(X_norm, y_norm, test_size=30,
                                                                            random_state=0)

    X_abnorm_train, X_abnorm_test, y_abnorm_train, y_abnorm_test = train_test_split(X_abnorm, y_abnorm, test_size=26,
                                                                                    random_state=0)

    X_train = np.vstack((X_norm_train, X_abnorm_train))
    y_train = np.r_[y_norm_train, y_abnorm_train]
    abnorm_train = len(y_abnorm_train) / len(y_train)
    X_test = np.vstack((X_norm_test, X_abnorm_test))
    y_test = np.r_[y_norm_test, y_abnorm_test]
    abnorm_test = len(y_abnorm_test)  # len(y_abnorm_test) / len(y_test)
    abnorm_train = len(y_abnorm_train)  # len(y_abnorm_train) / len(y_train)

    X_train_tuple = []
    for x in X_norm_train:
        X_train_tuple.append(tuple(x))
    y_train = y_norm
    # return X_train, X_test, y_train, y_test, abnorm_train, abnorm_test
    X_test_tuple = []
    for x in X:
        X_test_tuple.append(tuple(x))
    y_test = y
    print("X_train:%d,X_test:%d, y_train:%d, y_test:%d, abnorm_train:%f, abnorm_test:%f" % (
        len(X_train_tuple), len(X_test_tuple), len(y_train), len(y_test), abnorm_train, abnorm_test))
    return X_train_tuple, X_test_tuple, y_train, y_test, abnorm_train, abnorm_test


def pkl(file_dir):
    """
    将特征和标签写入pkl文件
    :param file_dir: 图片数据集路径
    :return:
    """
    files = []
    # get files at the current directory path
    for root, dirs, files_name in os.walk(file_dir):
        for file in files_name:
            files.append(root + '\\' + file)

    # 提取颜色矩特征
    feature_train = []
    for f in files:
        img = cv2.imread(f)
        ft_color = color_moments(img)
        # ft_hog = hog_feature(img)
        feature_train.append(ft_color)

    X = np.array(feature_train)
    label = []
    for f in files:
        if f.split('\\')[3].split('_')[0] == 'normal':
            label.append(1)
        else:
            label.append(-1)
    y = np.array(label)
    r = []
    for i in range(len(X)):
        r.append(np.r_[X[i], y[i]])
    with open('feature.pkl', 'wb') as f:
        pickle.dump(r, f)


def hog_feature(image):
    """
    提取hog特征
    :param image:
    :return:
    """

    feature = ft.hog(image,  # input image
                     orientations=9)  # return HOG map
    return feature


# Compute low order moments(1,2,3)
def color_moments(image):
    """compute feature with color moment"""

    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv2.split(hsv)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean()) ** 3)
    s_skewness = np.mean(abs(s - s.mean()) ** 3)
    v_skewness = np.mean(abs(v - v.mean()) ** 3)
    h_thirdMoment = h_skewness ** (1. / 3)
    s_thirdMoment = s_skewness ** (1. / 3)
    v_thirdMoment = v_skewness ** (1. / 3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])
    return color_feature


def create_data(file_dir, test_size):
    """
    通过计算颜色矩特征返回带有特征和标签的数据
    :param file_dir: 数据集文件路径
    :param test_size: 划分的测试集比例
    :return:
    """
    files = []
    # get files at the current directory path
    for root, dirs, files_name in os.walk(file_dir):
        for file in files_name:
            files.append(root + '\\' + file)

    # 提取颜色矩特征
    feature_train = []
    for f in files:
        img = cv2.imread(f)
        ft_color = color_moments(img)
        # ft_hog = hog_feature(img)
        feature_train.append(ft_color)

    X = np.array(feature_train)
    label = []
    for f in files:
        if f.split('\\')[3].split('_')[0] == 'normal':
            label.append(1)
        else:
            label.append(-1)
    y = np.array(label)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)  # random_state=1:每次得到的随机数组都一样，# 0 or None:每次得到的随机数据不一样

    return X, y

