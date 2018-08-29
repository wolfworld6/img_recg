from hyperopt import tpe
from hpsklearn import HyperoptEstimator, any_classifier

import cv2
import numpy as np
import os
from skimage import feature as ft
# 使用交叉验证的方法，把数据集分为训练集和测试集
from sklearn.model_selection import train_test_split
'''
   @author:bqFirst
   date:2018-8p

'''


def hog_feature(image):

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
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])
    return color_feature


def create_data(file_dir, test_size):
    """
    :param file_dir: 数据集文件路径
    :param test_size: 划分的测试集比例
    :return:
    """
    files_train = []
    # get files at the current directory path
    for root, dirs, files_name in os.walk(file_dir):
        for file in files_name:
            files_train.append(root + '\\' + file)

    # 提取颜色矩特征
    feature_train = []
    for f in files_train:
        img = cv2.imread(f)
        ft_color = color_moments(img)
        # ft_hog = hog_feature(img)
        feature_train.append(ft_color)

    X = np.array(feature_train)
    label_train = []
    for f in files_train:
        if f.split('\\')[3].split('_')[0] == 'normal':
            label_train.append(1)
        else:
            label_train.append(0)
    y = np.array(label_train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)  # random_state=1:每次得到的随机数组都一样，# 0 or None:每次得到的随机数据不一样

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    # 训练数据特征及标签
    file_dir = "E:\Data\\train"
    # 得到数据集和测试集
    X_train, X_test, y_train, y_test = create_data(file_dir, 0.25)
    estim = HyperoptEstimator(classifier=any_classifier('clf'), algo=tpe.suggest, seed=0)
    estim.fit(X_train, y_train)
    print(estim.score(X_test, y_test))
    print(estim.best_model())
