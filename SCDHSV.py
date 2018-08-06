# -*- coding:utf-8 -*-

from PIL import Image
import cv2
import numpy as np
import os
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
    print(error, end=':')

    # error < 45 说明为正常图片
    if error < threshold:
        if filename == 'normal':
            print("判断为正常: 判断正确")
            flag = 1
            return flag
        else:
            print("判断为正常: 判断错误")
            return flag
    else:
        if filename == 'abnormal':
            print("判断为异常: 判断正确")
            flag = 1
            return flag
        else:
            print("判断为异常: 判断错误")
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
    threshold = 1.5*dists[int(len(dists)/2)]
    print("threshold: ", threshold)
    return threshold


if __name__ == "__main__":

    file_dir_norm = "E:\eshore\PycharmProjects\Monitor_cam2\data1\lijiaoqiao\\normal"
    files_norm = []
    # get files at the current directory path
    for root, dirs, files_name in os.walk(file_dir_norm):
        for file in files_name:
            files_norm.append(root + '\\' + file)
    # compute features of normal images
    f_norm = []
    blocks = []
    imgs_cf_norm = []
    # 将图像分块
    vx = 1008
    vy = 1008
    n_blk = 4*3
    for b in range(n_blk):
        blocks.append([])
    for file in files_norm:
        img_cf_norm = []
        res = cut(file, vx, vy)
        for b in range(len(res)):
            blk_cf = color_moments(res[b])
            blocks[b].append(blk_cf)
            img_cf_norm.append(blk_cf)

        imgs_cf_norm.append(img_cf_norm)

    # 计算每个图像块特征的均值、标准差（所有normal图片）
    i = 0
    cf_mean_model = []  # 将正常图片的颜色矩特征的均值作为 model
    for b in blocks:
        # b：12 x 9 图片数量 x 特征维度
        i = i+1
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
    threh = train_threshold(imgs_cf_norm, cf_mean_model)
    # 根据标准差，得出比较活跃的块，并将这些块去掉

    # 计算测试图片的块特征，并于正常图片比对，然后进行异常判别
    file_dir_test = "E:\eshore\PycharmProjects\Monitor_cam2\data1\lijiaoqiao\\test"
    # compute features of test images
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

            print(file, end=': ')
            flag = abnorm_discriminant(file.split('_')[0], img_cf, cf_mean_model, threh)
            num_corr = num_corr + flag

    print("测试图片总数：%d, 正确判断总数：%d, 正确率为：%d" % (len(files_name), num_corr, num_corr/len(files_name)))


