# -*- coding:utf-8 -*-

import cv2
import numpy as np
import os
from skimage import feature as ft

from sklearn.model_selection import train_test_split
import pickle

from Davion import lof_tuning

'''
   @author:bqFirst
   date:2018-8
   孤立森林-K与threshold调优

'''


class PerformanceEvaluation:
    """
    算法性能评估
    :param TP: True Positive(真正)：将正类预测为正类数
    :param TN: True Negative(真负)：将负类预测为负类数
    :param FP: False Positive(假正)：将负类预测为正类数，误报
    :param FN: False Negative(假负)：将正类预测为负类数，漏报
    :param P: 预测为正类样本数
    :param N: 预测为负类样本数
    :return:
    """

    def __init__(self, tp, tn, fp, fn, p, n):
        self.__TP = tp
        self.__TN = tn
        self.__FP = fp
        self.__FN = fn
        self.__P = p
        self.__N = n

    def get(self):
        return self.__TP, \
               self.__TN, \
               self.__FP, \
               self.__FN, \
               self.__P, \
               self.__N

    def set(self, tp, tn, fp, fn, p, n):
        self.__TP = tp
        self.__TN = tn
        self.__FP = fp
        self.__FN = fn
        self.__P = p
        self.__N = n

    def get_accuracy(self):
        # 准确率
        acc = (self.__TP + self.__TN) / (self.__TP + self.__TN + self.__FP + self.__FN)
        return acc

    def get_error(self):
        # 错误率
        err = (self.__FP + self.__FN) / (self.__TP + self.__TN + self.__FP + self.__FN)
        return err

    def get_recall(self):
        # 灵敏度或召回率，所有正例中被分对的比率，衡量了分类器对正例的识别能力
        rec = self.__TP / self.__P
        return rec

    def get_specificity(self):
        # 特效度，所有负例中被分对的比例，衡量了分类器对负例的识别能力
        spec = self.__TN / self.__N
        return spec

    def get_precision(self):
        # 表示被分为正例的实例中实际为正例的比例
        p = self.__TP / (self.__TP + self.__FP)
        return p


def predict_labels(k, labels_predict, d):
    times = (6 - divmod(k, 6))[0]
    if d["t"] == 1.0:
        labels_predict[0+times*size].extend(d["label"])
    elif d["t"] == 1.2:
        labels_predict[1+times*size].extend(d["label"])
    elif d["t"] == 1.4:
        labels_predict[2+times*size].extend(d["label"])
    elif d["t"] == 1.6:
        labels_predict[3+times*size].extend(d["label"])
    elif d["t"] == 1.8:
        labels_predict[4+times*size].extend(d["label"])
    elif d["t"] == 2.0:
        labels_predict[5+times*size].extend(d["label"])
    elif d["t"] == 2.2:
        labels_predict[6+times*size].extend(d["label"])
    elif d["t"] == 2.4:
        labels_predict[7+times*size].extend(d["label"])
    elif d["t"] == 2.6:
        labels_predict[8+times*size].extend(d["label"])
    elif d["t"] == 2.8:
        labels_predict[9+times*size].extend(d["label"])
    elif d["t"] == 3.0:
        labels_predict[10+times*size].extend(d["label"])
    return labels_predict


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
    X_norm_train, X_norm_test, y_norm_train, y_norm_test = train_test_split(X_norm, y_norm, test_size=0.01,
                                                                            random_state=42)

    X_abnorm_train, X_abnorm_test, y_abnorm_train, y_abnorm_test = train_test_split(X_abnorm, y_abnorm, test_size=0.9,
                                                                                    random_state=42)
    X_train = np.vstack((X_norm_train, X_abnorm_train))
    y_train = np.r_[y_norm_train, y_abnorm_train]
    abnorm_train = len(y_abnorm_train) / len(y_train)
    X_test = np.vstack((X_norm_test, X_abnorm_test))
    y_test = np.r_[y_norm_test, y_abnorm_test]
    abnorm_test = len(y_abnorm_test) / len(y_test)
    abnorm_train = len(y_abnorm_train) / len(y_train)

    print("X_train:%d,X_test:%d, y_train:%d, y_test:%d, abnorm_train:%f, abnorm_test:%f" % (
        len(X_train), len(X_test), len(y_train), len(y_test), abnorm_train, abnorm_test))
    X_train_tuple = []
    for x in X_train:
        X_train_tuple.append(tuple(x))
    X_test_tuple = []
    for x in X_test:
        X_test_tuple.append(tuple(x))

    # return X_train, X_test, y_train, y_test, abnorm_train, abnorm_test
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


if __name__ == '__main__':

    rng = np.random.RandomState(42)
    file_dir = "E:\Data\\train"
    pklf = "./feature.pkl"
    X_train, X_test, y_train, y_test, outliers_fraction, unused = load(pklf)

    t = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    k = 5  # [5, 10, 15, 20 ,25, 30]

    valid = lof_tuning.lof_values(t, k, X_train)
    labels_predict = []
    for i in range(divmod(k, 5)[0]*len(t)):
        labels_predict.append([])
    size = len(t)
    for d in valid:
        if d["k"] == 5:
            print(d["k"])
            labels_predict = predict_labels(5, labels_predict, d)
        elif d["k"] == 10:
            labels_predict = predict_labels(10, labels_predict, d)
        elif d["k"] == 15:
            labels_predict = predict_labels(15, labels_predict, d)
        elif d["k"] == 20:
            labels_predict = predict_labels(20, labels_predict, d)
        elif d["k"] == 25:
            labels_predict = predict_labels(25, labels_predict, d)
        elif d["k"] == 30:
            labels_predict = predict_labels(30, labels_predict, d)

    for i in range(len(labels_predict)):
        labels = labels_predict[i]
        print(labels)
        # 计算预测正确率
        TP = 0  # 将正类预测为正类数
        TN = 0  # 将负类预测为负类数
        FP = 0  # 将负类预测为正类数,误报
        FN = 0  # 将正类预测为负类数,漏报
        P = 0  # 正样本数
        N = 0  # 负样本数
        for l in range(len(labels)):
            if labels[l] == 1:  # 预测为正类
                if labels[l] == y_train[l]:  # 实际为正类
                    P = P + 1
                    TP = TP + 1
                else:  # 实际为负类
                    FP = FP + 1
                    N = N + 1
            else:  # 预测为负类
                if labels[l] == y_train[l]:  # 实际为负类
                    TN = TN + 1
                    N = N + 1
                else:  # 实际为正类
                    P = P + 1
                    FN = FN + 1

        per_eva = PerformanceEvaluation(TP, TN, FP, FN, P, N)
        print("k: %d" % (divmod(i, 11)[0]+1)*5,
              "t: %f" % t[divmod(i, 6)[1]],
              "训练集正确率为：%f " % (per_eva.get_accuracy()),
              "召回率：%d/%d = %f" % (TP, P, per_eva.get_recall()),
              " 特效度：%d/%d = %f" % (TN, N, per_eva.get_specificity())
              )
