#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 16:11
# @Author  : Lijianfeng
# @Project : img_recg

from pHashModel.pHash import *
from itertools import combinations
import glob
import numpy as np
from pHashModel.phash_config import *

norm_cluster_imghash = {}   # {'cluster_name',[img_hash]}
norm_cluster_avg_dist = {}  # {'cluster_name', avg_distance}
abn_img_hash = {}

def img2hash(img_dir):
    """生成图片指纹
    :param img_dir:图片路径
    :return 图片指纹
    """
    os.chdir(img_dir)
    images = []
    images.extend(glob.glob('*.JPG'))
    img_hash = {}
    for i in images:
        img_hash[i] = avhash(i)

    return img_hash

def avg_dist(arr_hash):
    combins = [c for c in combinations(range(len(arr_hash)), 2)]
    hds = []    #N 张图片间相互的Hamming距离
    for i in combins:
        hm = hamming(arr_hash[i[0]], arr_hash[i[1]])
        hds.append(hm)
        # print(i, ":\t", hm, ":\t", arr_hash[i[0]], ":\t",  arr_hash[i[1]])
    avg_dist = np.mean(hds)
    return avg_dist

def normal_img_avg_dist(clusterd_img_dir):
    """N 张正常图片的平均 Hamming distance"""
    for root, dirs, files in os.walk(clusterd_img_dir):
        cluster_name = root.split("\\")[-1]
        avh = []
        for name in files:
            path = os.path.join(root, name)
            avh.append(avhash(path))
        if len(avh):
            norm_cluster_imghash[cluster_name] = avh
            norm_cluster_avg_dist[cluster_name] = avg_dist(avh)
    print(norm_cluster_imghash)
    print(norm_cluster_avg_dist)

def abnormal_hamming_distance():
    os.chdir(abnormal_img_dir)
    images = []
    images.extend(glob.glob('*.JPG'))
    for i in images:
        abn_img_hash[i] = avhash(i)

    # abn_img_dist = abn_img_hash.copy()
    for key, value in abn_img_hash.items():
        img_avg_dist = []
        for clu_name, img_hash in norm_cluster_imghash.items():
            distance = []
            for h in img_hash:
                ham = hamming(value, h)
                distance.append(ham)
            img_avg_dist.append(np.mean(distance))
            print(key, '距离簇心', clu_name, '的平均距离为: ', np.mean(distance))
        min_dist = np.min(img_avg_dist)
        print(key, " 与正常图片库的距离：", min_dist)

    return min_dist


if __name__ == '__main__':
    normal_img_avg_dist(clusterd_img_dir)
    abnormal_hamming_distance()


