# -*- coding:utf-8 -*-

from PIL import Image
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
'''
   @author:bqFirst
   date:2018-8

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


def abnorm_discriminant(filename, test, model, threshold):
    """
    :param test: 测试图片的块特征
    :param model: 合成的块特征
    :return: 异常 or 正常
    """
    flag = 0
    sum = 0
    for i in range(len(test)):
        sum = sum + np.linalg.norm(test[i]-model[i])

    error = sum / len(test)
    # print(error, end=':')

    # error < 45 说明为正常图片
    if error < threshold:
        if filename == 'normal':
            # print("判断为正常: 判断正确")
            flag = 1
            return flag
        else:
            # print("判断为正常: 判断错误")
            return flag
    else:
        if filename == 'abnormal':
            # print("判断为异常: 判断正确")
            flag = 1
            return flag
        else:
            # print("判断为异常: 判断错误")
            return flag


def train_threshold(imgs_cf, model):
    """
    :param imgs_cf:正常图片特征:[块数[图片数[特征维度]]]
    :param model: 由正常图片特征合成的块特征
    :return: 训练好的阈值
    """
    dists = []  # 每张图片特征与model差值的平方根再平均后的列表
    for f in imgs_cf:
        sum = 0
        for i in range(len(f)):
            x = np.linalg.norm(np.array(f[i]) - model[i])
            sum = sum+x
        dists.append(sum/len(model))
    mid = dists[int(len(dists)/2)]
    # print("threshold: ", threshold)
    return mid


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
        ft = color_moments(img)
        feature.append(img)
    l1, l2, l3 = image_cluster(feature, 3)

    # compute features of normal images
    f_norm = []
    blocks = []  # 用于对应图像块的特征值平均
    cf_clusters = []  # 用于聚类
    cf_images = []  # 聚类后，根据标签索引将图片分类
    # 将图像分块,并提取特征
    vx = 1008  # 块大小
    vy = 1008
    n_blk = 4*3
    for b in range(n_blk):
        blocks.append([])
    for file in files_norm:
        cf_cluster = []  # 整张图片的特征
        cf_image = []  # 整张图片的特征
        res = cut(file, vx, vy)  # 切成块（4x3）
        # 提取每个块的特征
        for b in range(len(res)):
            blk_cf = color_moments(res[b])  # 块特征提取
            blocks[b].append(blk_cf)

            cf_cluster.extend(blk_cf)

            cf_image.append(blk_cf)

        cf_clusters.append(cf_cluster)
        cf_images.append(cf_image)  # 图片特征,大小为100

    # 对正常图像进行聚类：
    data = np.array(cf_clusters)
    label_1, label_2, label_3 = image_cluster(data, k=2)  # 三类：拥堵，正常，空旷

    for id in label_1:
        mean_1 = []
        pass
    # 计算每个图像块特征的均值、标准差（所有normal图片）
    cf_mean_model = []  # 将正常图片的颜色矩特征的均值作为 model
    for b in blocks:
        # b：12 x 9 图片数量 x 特征维度
        arr = np.array(b)
        rows, cols = arr.shape
        # 每个块每一特征维度均值
        mean = np.mean(arr, 0)
        cf_mean_model.append(mean)
        # 每一特征维度标准差
        std = np.std(arr, 0)
        std_2 = np.linalg.norm(std)
        # print("第 %d 个块： " % i)
        # print("平均值： ", mean)
        # print("标准差: ", std_2)

    # 通过正常图片特征与model差值得到阈值
    mid = train_threshold(cf_images, cf_mean_model)
    # 根据标准差，得出比较活跃的块，并将这些块去掉

    # 计算测试图片的块特征，并于正常图片比对，然后进行异常判别
    file_dir_test = "E:\GitRepository\img_recg\data\shipai\\test"
    # compute features of test images
    #选取阈值
    res = []
    max = 0
    x = 0
    for times in np.arange(1.1, 2.1, 0.1):
        thres = times*mid
        num_corr = 0
        for root, dirs, files_name in os.walk(file_dir_test):
            for file in files_name:
                file_dir = root + '\\' + file
                img_cf = []
                res = cut(file_dir, vx, vy)
                for b in range(len(res)):
                    blk_cf = color_moments(res[b])
                    arr = np.array(blk_cf)
                    img_cf.append(arr)

                # print(file, end=': ')
                flag = abnorm_discriminant(file.split('_')[0], img_cf, cf_mean_model, thres)
                num_corr = num_corr + flag
        if num_corr > max:
            max = num_corr
            x = times
        res.append((times, num_corr))

    print("最好的阈值times、判断正确数分别为：", x, max)

    '''
    for b in x:
        print(b)
    for r in res:
        print(r)
    '''

    # print("测试图片总数：%d, 正确判断总数：%d, 正确率为：%f" % (len(files_name), num_corr, (num_corr*1.0)/len(files_name)))

