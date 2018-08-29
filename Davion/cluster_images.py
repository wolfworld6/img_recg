# -*- coding:utf-8 -*-

from PIL import Image
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
'''
   @author:bqFirst
   date:2018-8
   对图像进行聚类
'''


# Compute low order moments(1,2,3)
def color_moments(img):
    """compute feature with color moment"""

    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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


def color_mean(img):
    """compute feature with color moment"""

    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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

    return color_feature


def cut(file,vx,vy):
    """
    :param file:
    :param vx: 块大小
    :param vy: 块大小
    :return:分块集
    像素大小：4032 x 3024 分为：12 x 9个块 阈值:1.5*中位数(根据训练样本的变化调整)
                        分为：4 x 3个块 正确率为：0.95左右(比12x9稍高)
    """
    im = Image.open(file)

    #偏移量,滑动量
    dx = vx
    dy = vy
    n = 1

    #左上角切割
    x1 = 0
    y1 = 0
    x2 = vx
    y2 = vy

    imgs = []
    #纵向
    while x2 <= 3024:
        #横向切
        while y2 <= 4032:
            name3 = file + str(n) + ".jpg"
            #print n,x1,y1,x2,y2
            im2 = im.crop((y1, x1, y2, x2))
            im2_mat = np.asanyarray(im2)
            imgs.append(im2_mat)
            # im2.save(name3)

            y1 = y1 + dy
            y2 = y1 + vy
            n = n + 1
        x1 = x1 + dx
        x2 = x1 + vx
        y1 = 0
        y2 = vy

    return imgs


def image_cluster(data, k):
    """
    :param data:100 * 108（12x9）
    :param k: 我们将图片分成三类，大概为：拥堵，正常，空旷
    :return: 类别索引
    """
    estimator = KMeans(n_clusters=k,init='k-means++')
    estimator.fit(data)
    label_pred = estimator.labels_
    label_1 = []
    label_2 = []
    label_3 = []
    for l in range(len(label_pred)):
        if label_pred[l] == 0:
            label_1.append(l)
        elif label_pred[l] == 1:
            label_2.append(l)
        else:
            label_3.append(l)

    centroids = estimator.cluster_centers_
    print("标签：\n", label_pred)
    # print("聚类中心：\n",centroids)
    return label_1, label_2, label_3


if __name__ == "__main__":

    file_dir_norm = "E:\GitRepository\img_recg\data\shipai\\train"
    files_norm = []
    # get files at the current directory path
    for root, dirs, files_name in os.walk(file_dir_norm):
        for file in files_name:
            files_norm.append(root + '\\' + file)

    # 提取整张图片特征，用于聚类
    feature = []
    for f in files_norm:
        img = cv2.imread(f)
        ft = color_mean(img)
        feature.append(ft)
    l1, l2, l3 = image_cluster(feature, 3)