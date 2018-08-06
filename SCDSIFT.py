"""
monitor detection
"""
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import os


def get_sift(img, out_file):
    """get the sift features with image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    # img = cv2.drawKeypoints(gray, key_point, img)
    # cv2.imwrite(out_file, img)

    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyWindow('image')
    return kp, des


def get_surf(in_file, out_file):
    """get the surf feature with image"""
    img = cv2.imread(in_file, 0)
    surf = cv2.xfeatures2d.SURF_create(400)
    surf.setExtended(True)
    kp, des = surf.detectAndCompute(img, None)
    # img2 = cv2.drawKeypoints(img, key_point, None, (255, 0, 0), 4)
    # cv2.imwrite(out_file, img2)
    return kp, des


def occlusion_discriminant():
    """the discriminant rule of occlusion"""
    pass


def skew_discriminant():
    """the discriminant rule of skew"""
    pass


def black_screen_discriminant():
    """the discriminant rule of black screen"""
    pass


def statistical_figure(null):
    """the statistical of the mean of shift or the number of matched key points"""

    '''
    N = 5
    ind = np.arange(N)
    width = 0.35
    menMeans = (20, 35, 30, 35, 27)
    menStd = (2, 3, 4, 1, 2)
    womenMeans = (25, 32, 34, 20, 25)
    womenStd = (3, 5, 2, 3, 3)
    plt.bar(ind, menMeans, width, color='r', yerr=menStd)
    plt.bar(ind+width, womenMeans, width, color='y', yerr=womenStd)
    plt.title('Men-Women')
    plt.show()
    '''

    x1 = 10 + 5 * np.random.randn(10000)

    x2 = 20 + 5 * np.random.randn(10000)

    num_bins = 50

    plt.hist(x1, num_bins, density=1, facecolor='green', alpha=0.5)
    plt.hist(x2, num_bins, density=1, facecolor='blue', alpha=0.5)

    plt.title('Histogram')

    plt.show()


def sift_match(image1: [], image2: []):
    """match two images based on SIFT ,
     return the number of matched key points
     and the shift
     and the ratio of matched
    """
    # print('the original image shape: ', img1.shape)

    # generate Gaussian pyramid for img1
    G1 = image1.copy()
    gpKP1 = [G1]
    for i in range(3):
        G1 = cv2.pyrDown(G1)
        gpKP1.append(G1)

    # generate Gaussian pyramid for img2
    G2 = image2.copy()
    gpKP2 = [G2]
    for i in range(3):
        G2 = cv2.pyrDown(G2)
        gpKP2.append(G2)

    img_G1 = gpKP1[2].copy()
    img_G2 = gpKP2[2].copy()

    # print('the Gaussian pyramid image shape: ', img_G1.shape)
    cv2.imwrite('img_G1.jpg', img_G1)
    cv2.imwrite('img_G2.jpg', img_G2)

    time_start = time.time()
    # find the key points and descriptors with SIFT
    kp1, des1 = get_sift(img_G1, None)
    kp2, des2 = get_sift(img_G2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(check=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # find k nearest neighbors
    matches1 = flann.knnMatch(des2, des1, k=2)

    # TO find good matches, we have two methods:(1)ratio test;(2)match each other

    # ratio test to find good matches
    good = []
    for m, n in matches1:
        if m.distance < 0.7 * n.distance:
            good.append([m])

    '''
    # match each other to find good matches
    matches2 = flann.knnMatch(des1, des2, k=2)
    good = match_each_other(kp1, matches1, kp2, matches2)
    '''
    # matched points' position detection
    x_shift = []
    y_shift = []
    for i in range(len(good)):
        queryIdx = good[i][0].queryIdx
        trainIdx = good[i][0].trainIdx

        # find the position with pixel
        x2 = np.round(kp2[queryIdx].pt[0])
        y2 = np.round(kp2[queryIdx].pt[1])

        x1 = np.round(kp1[trainIdx].pt[0])
        y1 = np.round(kp1[trainIdx].pt[1])

        # compute the pixel's difference between matched key points
        x_shift.append(abs(x1 - x2))
        y_shift.append(abs(y1 - y2))

    # the number of good matches
    # print('the number of good matches, kp1, kp2: ', len(good), len(kp1), len(kp2))
    # print('matched key points / the total kp1:', len(good) / len(kp1))
    # print('matched key points / the total kp2:', len(good) / len(kp2))

    # compute the shift between the normal and the others
    sum_x = 0
    sum_y = 0
    for i in x_shift:
        sum_x += i
    for i in y_shift:
        sum_y += i

    x_shift_mean = sum_x / len(x_shift)
    y_shift_mean = sum_y / len(y_shift)

    shift = np.sqrt((pow(x_shift_mean, 2)+pow(y_shift_mean, 2)))
    # print('the average shift in the X direction:', x_shift_mean)
    # print('the average shift in the Y direction:', y_shift_mean)

    time_end = time.time()
    print('knnMatch cost time: ', time_end - time_start)

    matchColor = (0, 255, 0)
    singlePointColor = (255, 0, 0)

    #img3 = cv2.drawMatchesKnn(img_G1, kp1, img_G2, kp2, matches, None,
     #                         matchColor, singlePointColor, matchesMask, flags)
    img3 = cv2.drawMatchesKnn(img_G1, kp1, img_G2, kp2, good, None,
                              matchColor, singlePointColor)
    cv2.imwrite('sift_matches.jpg', img3)
    #plt.imshow(img3)
    #plt.show()
    # return the number of good matches, the shift of matched key points, the ratios of matched
    return len(good), shift, (len(good) / len(kp1)+len(good) / len(kp2))/2


def match_each_other(kp1: list, match1: list, kp2: list, match2: list):
    """Get the matches of which match each other"""
    m_each_other = []
    for i in range(len(match1)):
        kp1_Idx = match1[i][0].queryIdx
        kp2_Idx = match1[i][0].trainIdx

        if match2[kp2_Idx][0].trainIdx is kp1_Idx:
           m_each_other.append(match1[i])
    return m_each_other


if __name__ == '__main__':

    file_dir_norm = "M:\eshore\PycharmProjects\Monitor_cam2\\normal\\normal2"
    files_norm = []
    # get files at the current directory path
    for root, dirs, files_name in os.walk(file_dir_norm):
        for file in files_name:
            files_norm.append(root+'\\'+file)
    # the match in normal images
    shifts_norm = []
    ratios_norm = []
    for file1 in files_norm:
        for file2 in files_norm:
            if file1 is not file2:
                img1 = cv2.imread(file1, cv2.IMREAD_COLOR)
                img2 = cv2.imread(file2, cv2.IMREAD_COLOR)
                match_number, shift, ratio = sift_match(img1, img2)
                shifts_norm.append(shift)
                ratios_norm.append(ratio)

    # read the images of test
    file_dir_test = "M:\eshore\PycharmProjects\Monitor_cam2\\new"
    files_test = []
    for root, dirs, files_name in os.walk(file_dir_test):
        for file in files_name:
            files_test.append(root+'\\'+file)

    # the match between test image and normal images
    shifts_test = []
    ratios_test = []
    for file1 in files_test:
        for file2 in files_norm:
            img1 = cv2.imread(file1, cv2.IMREAD_COLOR)
            img2 = cv2.imread(file2, cv2.IMREAD_COLOR)
            match_number, shift, ratio = sift_match(img1, img2)
            shifts_test.append(shift)
            ratios_test.append(ratio)

    plt.hist(np.array(shifts_norm), 5, density=1, facecolor='green', alpha=0.5)
    plt.hist(np.array(shifts_test), 5, density=1, facecolor='blue', alpha=0.5)

    plt.title('Histogram')

    plt.show()

