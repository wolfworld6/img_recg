import os
from pHash import *
from itertools import combinations
import glob
import numpy as np

# normal_img_dir = 'E:\\bigdata\\pycharmProject\\image_recognition\\data\\2018_07_01\\normal'
# abnormal_img_dir = 'E:\\bigdata\\pycharmProject\\image_recognition\\data\\2018_07_01\\abnormal'
normal_img_dir = 'E:\\bigdata\\pycharmProject\\image_recognition\\data\\shipai\\normal'
abnormal_img_dir = 'E:\\bigdata\\pycharmProject\\image_recognition\\data\\shipai\\abnormal'
norm_img_hash = []
abn_img_hash = {}

def normal_img_avg_dist():
    """N 张正常图片的平均 Hamming distance"""
    os.chdir(normal_img_dir)
    images = []
    images.extend(glob.glob('*.JPG'))
    for i in images:
        norm_img_hash.append(avhash(i))

    combins = [c for c in combinations(range(len(norm_img_hash)), 2)]
    hds = []    #N 张图片间相互的Hamming距离

    for i in combins:
        hds.append(hamming(norm_img_hash[i[0]], norm_img_hash[i[1]]))

    avg_dist = np.mean(hds)
    # avg_dist = np.median(hds)
    print(len(images), "张图片相互间平均汉明距离：",  avg_dist)

    return avg_dist

def abnormal_hamming_distance():
    os.chdir(abnormal_img_dir)
    images = []
    images.extend(glob.glob('*.JPG'))
    for i in images:
        abn_img_hash[i] = avhash(i)

    abn_img_dist = abn_img_hash.copy()
    for key, value in abn_img_hash.items():
        distance = []
        for norm in norm_img_hash:
            h = hamming(value, norm)
            distance.append(h)
        abn_img_dist[key] = np.mean(distance)
        print(key, " 与正常图片库的平均距离：", np.mean(distance))

    return abn_img_dist

def kmeans_cluster():
    """对图片进行聚类"""



if __name__ == '__main__':
    normal_img_avg_dist()
    abnormal_hamming_distance()


