# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 16:11
# @Author  : Lijianfeng
# @Project : img_recg

from pHashModel.pHash import *
from pHashModel.phash_config import *
import os
import pandas as pd
from sklearn.model_selection import train_test_split

img_labels = []


def init_img():
    """
    将图片转换成指纹，并打上标签，保存到  knn_data_set.csv 中
    :return:
    """
    # df_clu_img(clusterd_img_dir)
    img_label(normal_img_dir, 'nor')
    img_label(abnormal_img_dir, 'abn')
    os.chdir(sys.path[0])
    pd.DataFrame(img_labels, columns=["img", "fingerprint", "label"]).to_csv(knn_data_set)


def df_clu_img(clusterd_img_dir):
    """
    把clusterd中的图片文件转换成指纹
    :param img_dir:
    :return:
    """
    # 读取聚类后的图片
    for root, dirs, files in os.walk(clusterd_img_dir):
        # cluster_name = root.split("\\")[-1]
        for name in files:
            path = os.path.join(root, name)
            avh = avhash(path)
            img_labels.append([name, avh, 'nor'])

    return img_labels


def img_label(abnormal_img_dir, label):
    """
    通用的图片转换成指纹
    :param abnormal_img_dir:
    :return:
    """
    os.chdir(sys.path[0])
    os.chdir(abnormal_img_dir)
    images = []
    images.extend(glob.glob('*.JPG'))
    for i in images:
        img_labels.append([i, avhash(i), label])

    return img_labels


def split_train_test(df):
    X_train, X_test, y_train, y_test = train_test_split(df, df, test_size=0.3, random_state=0)
    return X_train, X_test


def knn(k, train, test):
    """
    实现 knn 算法
    :param k:
    :param train:
    :param test:
    :return:
    """
    result = []
    for test_index, test_row in test.iterrows():
        h = []
        for train_index, train_row in train.iterrows():
            h.append([test_row['img'], train_row['img'], hamming(test_row['fingerprint'], train_row['fingerprint']), test_row['label'], train_row['label']])
        dfh = pd.DataFrame(h, columns=['test_img', 'train_img', 'hamming', 'test_label', 'train_label'])
        dfh = dfh.sort_values(by='hamming')
        dfh = dfh[:k]
        predict_label = dfh['train_label'].value_counts().idxmax()
        result.append([test_row['img'], test_row['label'], predict_label])
    return pd.DataFrame(result, columns=['img', 'original_label', 'predict_label'])   # 预测结果


def summary(predict_result):
    tp = predict_result[(predict_result['original_label'] == 'abn') & (predict_result['predict_label'] == 'abn')].iloc[:, 0].size
    fn = predict_result[(predict_result['original_label'] == 'abn') & (predict_result['predict_label'] == 'nor')].iloc[:, 0].size
    fp = predict_result[(predict_result['original_label'] == 'nor') & (predict_result['predict_label'] == 'abn')].iloc[:, 0].size
    tn = predict_result[(predict_result['original_label'] == 'nor') & (predict_result['predict_label'] == 'nor')].iloc[:, 0].size

    a = (tp + tn) / (tp + fn + fp + tn)     # 正确率accuracy：正确预测的
    p = tp / (tp + fp)  # 精确率precision：预测为异常图片中多少正确的
    r = tp / (tp + fn)  # 召回率recall：总共的异常图片中多少被正确预测为异常的
    f1 = 2 * (p * r/(p + r))    # F1-measure：精确值和召回率的调和均值，精确率和召回率都高时，F1值也会高
    return a, p, r, f1


def main():
    # 初始化数据时执行 init_img()
    # init_img()
    df = pd.read_csv(knn_data_set)
    train, test = split_train_test(df)
    predict_result = knn(8, train, test)
    a, p, r, f1 = summary(predict_result)
    row = pd.DataFrame([['accuracy', '', a], ['precision', '', p], ['recall', '', r], ['F1-measure', '', f1]], columns=['img', 'original_label', 'predict_label'])
    predict_result = predict_result.append(row, ignore_index=True)
    os.chdir(sys.path[0])
    predict_result.to_csv(knn_result)
    print(predict_result)
    print('accuracy: ', a, 'precision: ', p, '\t', 'recall', r, '\t', 'F1-measure:', f1)


if __name__ == '__main__':
    main()
