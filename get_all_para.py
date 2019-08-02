#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
import skimage
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn import mixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
import cv2
import math
import random



def Ellipse_fit(x, y):
    # a*x**2 + b*x*y + c*y**2 + d*x + e*y + f
    x0, y0 = x.mean(), y.mean()
    D1 = np.array([(x - x0) ** 2, (x - x0) * (y - y0), (y - y0) ** 2]).T
    D2 = np.array([x - x0, y - y0, np.ones(y.shape)]).T
    S1 = np.dot(D1.T, D1)
    S2 = np.dot(D1.T, D2)
    S3 = np.dot(D2.T, D2)
    T = -1 * np.dot(np.linalg.inv(S3), S2.T)
    M = S1 + np.dot(S2, T)
    M = np.array([M[2] / 2, -M[1], M[0] / 2])
    lam, eigen = np.linalg.eig(M)
    cond = 4 * eigen[0] * eigen[2] - eigen[1] ** 2
    Q1 = eigen[:, cond > 0]
    Q = np.vstack([Q1, np.dot(T, Q1)]).flatten()
    Q3 = Q[3] - 2 * Q[0] * x0 - Q[1] * y0
    Q4 = Q[4] - 2 * Q[2] * y0 - Q[1] * x0
    Q5 = Q[5] + Q[0] * x0 ** 2 + Q[2] * y0 ** 2 + Q[1] * x0 * y0 - Q[3] * x0 - Q[4] * y0
    Q[3] = Q3
    Q[4] = Q4
    Q[5] = Q5

    # 计算标准椭圆中心位置，长短轴和长轴倾角
    paras = Q / Q[5]
    A, B, C, D, E = paras[:5]
    x0 = (B * E - 2 * C * D) / (4 * A * C - B ** 2)
    y0 = (B * D - 2 * A * E) / (4 * A * C - B ** 2)
    a = 2 * np.sqrt((2 * A * (x0 ** 2) + 2 * C * (y0 ** 2) + 2 * B * x0 * y0 - 2) / (A + C + np.sqrt(((A - C) ** 2 + B ** 2))))
    b = 2 * np.sqrt((2 * A * (x0 ** 2) + 2 * C * (y0 ** 2) + 2 * B * x0 * y0 - 2) / (A + C - np.sqrt(((A - C) ** 2 + B ** 2))))
    q = 0.5 * np.arctan(B / (A - C))
    return x0, y0, a, b, q


def Gaussian_fit(x, y, value):
    C = np.array([np.sum(x ** 2 * np.log(value)),
                  np.sum(y ** 2 * np.log(value)),
                  np.sum(x * np.log(value)),
                  np.sum(y * np.log(value)),
                  np.sum(np.log(value))])

    B = np.array([[np.sum(x ** 4), np.sum(x ** 2 * y ** 2), np.sum(x ** 3), np.sum(x ** 2 * y), np.sum(x ** 2)],
                  [np.sum(x ** 2 * y ** 2), np.sum(y ** 4), np.sum(x * y ** 2), np.sum(y ** 3), np.sum(y ** 2)],
                  [np.sum(x ** 3), np.sum(x * y ** 2), np.sum(x ** 2), np.sum(x * y), np.sum(x)],
                  [np.sum(x ** 2 * y), np.sum(y ** 3), np.sum(x * y), np.sum(y ** 2), np.sum(y)],
                  [np.sum(x ** 2), np.sum(y ** 2), np.sum(x), np.sum(y), len(value)]])

    B_inv = np.linalg.inv(B)
    K = B_inv.dot(C.T)
    # print("C:\n", C)
    # print("B:\n", B)
    # print("B_inv:\n", B_inv)
    # print("K:\n", K)
    # print("len(value):\n", len(x), len(y), len(value))
    # K[0] = -abs(K[0])
    # K[1] = -abs(K[1])
    # K[2] = abs(K[2])
    # K[3] = abs(K[3])
    return K


def luminanace(r, g, b):
    a = np.array(list(map(lambda x: x / 12.92 if x <= 0.03928 else pow((x + 0.055) / 1.055, 2.4), [r, g, b])))
    c = np.array([0.2126, 0.7152, 0.0722])
    return np.sum(a * c)


def contrast(rgb1, rgb2):
    return (luminanace(rgb1[0], rgb1[1], rgb1[2]) + 0.05) / (luminanace(rgb2[0], rgb2[1], rgb2[2]) + 0.05)


def Gaussian_fit_rotation(x, y, value):
    best_score = 999999999
    best_angle = None
    for i in range(0, 100):
        # 创建随机解
        angle = random.uniform(0, 90)
        radian = angle * np.pi / 180

        x_new = np.array(x) * np.cos(radian) - np.array(y) * np.sin(radian)
        y_new = np.array(x) * np.sin(radian) + np.array(y) * np.cos(radian)
        M = Gaussian_fit(x_new, y_new, np.array(value))

        # 计算成本值
        current_score = 1 / (-2 * M[0]) + 1 / (-2 * M[1])
        # cost = costf(x_A, y_A, value_A, angle * np.pi / 180)
        # 与目前得到的最优解进行比较
        if current_score < best_score:
            best_score = current_score
            best_angle = angle
            K = M

    K = np.append(K, best_angle)

    return K



if __name__ == '__main__':

    dataframe = pd.DataFrame(columns=['distance', 'sigma_x_A', 'sigma_y_A', 'sigma_x_B', 'sigma_y_B'])


    tgt_layer = 'convolution_7'
    units = list(range(0, 512))
    u_num = len(units)
    # unit_use = 6

    data_all = np.zeros((u_num, 128, 128, 3))
    data_black = np.zeros((128, 128, 3))
    for i in range(u_num):
        data_all[i] = pd.read_pickle("./visualize_laplacian/visualize-lap-{}-{}-15.pkl".format(tgt_layer, units[i]))[64:192, 64:192]


    # print("All shape: ", data_all.shape)
    for unit_use in range(0, 512):
        print("unit_use", unit_use)
        data_row = data_all[unit_use]
        # print(data_row[:, :, 0])


        # plt.figure()
        # plt.subplot(4, 4, 1)
        # plt.axis('off')
        # data_RGB = np.uint8(np.clip(data_row, 0, 1) * 255.0)
        # plt.imshow(data_RGB)


        # 先转化为二维数据
        image_array = data_row.reshape((data_row.shape[0] * data_row.shape[1], 3))
        # print(image_array.shape)

        # 1st: GMM
        clst = mixture.GaussianMixture(n_components=3, max_iter=100, covariance_type="full")
        clst.fit(image_array)
        predicted_labels = clst.predict(image_array)
        centroids_GMM = clst.means_
        # print("centroids_GMM", centroids_GMM)
        labels = predicted_labels
        label_matrix = labels.reshape((128, 128))



        # GMM处理后的三类分别黑白显示出来
        data_gmm_0 = data_row.copy()
        for i in range(128):
            for j in range(128):
                if label_matrix[i, j] == 0:
                    data_gmm_0[i][j] = [1, 1, 1]
                else:
                    data_gmm_0[i][j] = [0, 0, 0]
        # plt.subplot(4, 4, 2)
        # plt.axis('off')
        # data_0 = np.uint8(np.clip(data_gmm_0, 0, 1) * 255.0)
        # plt.imshow(data_0)

        data_gmm_1 = data_row.copy()
        for i in range(128):
            for j in range(128):
                if label_matrix[i, j] == 1:
                    data_gmm_1[i][j] = [1, 1, 1]
                else:
                    data_gmm_1[i][j] = [0, 0, 0]
        # plt.subplot(4, 4, 3)
        # plt.axis('off')
        # data_1 = np.uint8(np.clip(data_gmm_1, 0, 1) * 255.0)
        # plt.imshow(data_1)

        data_gmm_2 = data_row.copy()
        for i in range(128):
            for j in range(128):
                if label_matrix[i, j] == 2:
                    data_gmm_2[i][j] = [1, 1, 1]
                else:
                    data_gmm_2[i][j] = [0, 0, 0]
        # plt.subplot(4, 4, 4)
        # plt.axis('off')
        # data_2 = np.uint8(np.clip(data_gmm_2, 0, 1) * 255.0)
        # plt.imshow(data_2)


        # 找中心的那一块， 求出最大值的索引
        baseline = np.array([0.5] * 9).reshape(3, 3)

        max_index = np.argmax(np.sum(abs(centroids_GMM - baseline), axis=1))

        # 从原始图像中将除中心区域外的区域变黑/灰/抠掉
        data_center = data_row.copy()
        if max_index == 0:
            for i in range(128):
                for j in range(128):
                    if label_matrix[i, j] != 0:
                        data_center[i][j] = [0.5, 0.5, 0.5]  # [0.5, 0.5, 0.5]  # [0, 0, 0]  # [np.nan, np.nan, np.nan] # 四周平均

        elif max_index == 1:
            for i in range(128):
                for j in range(128):
                    if label_matrix[i, j] != 1:
                        data_center[i][j] = [0.5, 0.5, 0.5]
        else:
            for i in range(128):
                for j in range(128):
                    if label_matrix[i, j] != 2:
                        data_center[i][j] = [0.5, 0.5, 0.5]

        # plt.subplot(4, 4, 5)
        # plt.axis('off')
        data_center = np.clip(data_center, 0, 1)
        # # print("data_center:\n", data_center)
        # plt.imshow(data_center)



        # 处理后的图像再次展开为二维数据
        image_array_new = data_center.reshape((data_row.shape[0] * data_row.shape[1], 3))


        # 2nd: k-means聚类
        kmeans = KMeans(n_clusters=3,  init='k-means++')
        kmeans.fit(image_array_new)
        centroids_kmeans = kmeans.cluster_centers_
        labels = kmeans.labels_
        # print("centroids_k-means", centroids_kmeans)
        label_matrix_new = labels.reshape((128, 128))


        # 上步在处理过后的中心数据集上进行k-means后分成了三类，这里分别黑白显示
        data_0_new = data_center.copy()
        for i in range(128):
            for j in range(128):
                if label_matrix_new[i, j] == 0:
                    data_0_new[i][j] = [1, 1, 1]
                else:
                    data_0_new[i][j] = [0, 0, 0]
        # plt.subplot(4, 4, 6)
        # plt.axis('off')
        # plt.imshow(data_0_new)

        data_1_new = data_center.copy()
        for i in range(128):
            for j in range(128):
                if label_matrix_new[i, j] == 1:
                    data_1_new[i][j] = [1, 1, 1]
                else:
                    data_1_new[i][j] = [0, 0, 0]
        # plt.subplot(4, 4, 7)
        # plt.axis('off')
        # plt.imshow(data_1_new)

        data_2_new = data_center.copy()
        for i in range(128):
            for j in range(128):
                if label_matrix_new[i, j] == 2:
                    data_2_new[i][j] = [1, 1, 1]
                else:
                    data_2_new[i][j] = [0, 0, 0]
        # plt.subplot(4, 4, 8)
        # plt.axis('off')
        # plt.imshow(data_2_new)



        # 找到中间需要用那两类
        min_index = np.argmin(np.sum(abs(centroids_kmeans - baseline), axis=1))

        if min_index == 0:
            data_A = data_1_new
            data_B = data_2_new
            class_A = 1
            class_B = 2
            avg_A = centroids_kmeans[1]
            avg_B = centroids_kmeans[2]
        elif min_index == 1:
            data_A = data_0_new
            data_B = data_2_new
            class_A = 0
            class_B = 2
            avg_A = centroids_kmeans[0]
            avg_B = centroids_kmeans[2]
        else:
            data_A = data_0_new
            data_B = data_1_new
            class_A = 0
            class_B = 1
            avg_A = centroids_kmeans[0]
            avg_B = centroids_kmeans[1]


        # 膨胀与腐蚀，去掉离散点
        kernel = np.ones((5, 5), np.uint8)
        data_A = cv2.morphologyEx(cv2.morphologyEx(data_A, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
        # plt.subplot(4, 4, 13)
        # plt.axis('off')
        # plt.imshow(data_A)

        data_B = cv2.morphologyEx(cv2.morphologyEx(data_B, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
        # plt.subplot(4, 4, 14)
        # plt.axis('off')
        # plt.imshow(data_B)



        # OpenCV找外接椭圆的方法（必须用在腐蚀操作后，否则会提取出好多小椭圆，这样也不能完全避免，取其中参数长度最长的，应该也就是最大的圆）
        x0_A, y0_A, a_A, b_A, angle_A, x0_B, y0_B, a_B, b_B, angle_B = (0 for _ in range(10))
        img_merge = 0
        max_cnt = 0
        imgray = cv2.cvtColor(np.uint8(np.clip(data_A, 0, 1) * 255), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt_index, cnt in enumerate(contours):
            # print("cnt_index, len(cnt): ", cnt_index, "\t", len(cnt))
            if len(cnt) > max_cnt:
                max_cnt = len(cnt)
                max_cnt_index = cnt_index
        if max_cnt > 5:
            ellipse = cv2.fitEllipse(contours[max_cnt_index])
            x0_A, y0_A, a_A, b_A, angle_A = ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], ellipse[2]
            # print("ellipse_A_parameters: ", x0_A, y0_A, a_A, b_A, angle_A)
            # img_tmp = cv2.ellipse(data_A, ellipse, (1, 0, 0), 2)
            # img_merge = cv2.ellipse(data_black, ellipse, (1, 0, 0), 2)
            # plt.subplot(4, 4, 11)
            # plt.axis('off')
            # plt.imshow(img_tmp)

        max_cnt = 0
        imgray = cv2.cvtColor(np.uint8(np.clip(data_B, 0, 1) * 255), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt_index, cnt in enumerate(contours):
            if len(cnt) > max_cnt:
                max_cnt = len(cnt)
                max_cnt_index = cnt_index
        if max_cnt > 5:
            ellipse = cv2.fitEllipse(contours[max_cnt_index])
            x0_B, y0_B, a_B, b_B, angle_B = ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], ellipse[2]
            # print("ellipse_B_parameters: ", x0_B, y0_B, a_B, b_B, angle_B)
            img_tmp = cv2.ellipse(data_B, ellipse, (0, 0, 1), 2)
            img_merge = cv2.ellipse(img_merge, ellipse, (0, 0, 1), 2)
            # plt.subplot(4, 4, 12)
            # plt.axis('off')
            # plt.imshow(img_tmp)
            # plt.subplot(4, 4, 15)
            # plt.axis('off')
            # plt.imshow(img_merge)



        # 转成灰度图后进行二次高斯拟合
        x_A, y_A, value_A, x_B, y_B, value_B = ([] for _ in range(6))
        for i in range(128):
            for j in range(128):
                if np.sum(data_center[i, j]) != 0:
                    if label_matrix_new[i, j] == class_A:
                        x_A.append(i)
                        y_A.append(j)
                        value_A.append((data_center[i][j][0] * 0.2989 + data_center[i][j][1] * 0.5870 + data_center[i][j][2] * 0.1140))
                        # value_A.append(1)
                    else:
                        x_A.append(i)
                        y_A.append(j)
                        value_A.append(0.01)
                    if label_matrix_new[i, j] == class_B:
                        x_B.append(i)
                        y_B.append(j)
                        value_B.append((data_center[i][j][0] * 0.2989 + data_center[i][j][1] * 0.5870 + data_center[i][j][2] * 0.1140))
                        # value_B.append(1)
                    else:
                        x_B.append(i)
                        y_B.append(j)
                        value_B.append(0.01)


        K_A = Gaussian_fit(np.array(x_A), np.array(y_A), np.array(value_A))
        K_B = Gaussian_fit(np.array(x_B), np.array(y_B), np.array(value_B))

        K_A_ro = Gaussian_fit_rotation(np.array(x_A), np.array(y_A), np.array(value_A))
        K_B_ro = Gaussian_fit_rotation(np.array(x_B), np.array(y_B), np.array(value_B))

        x0gA = K_A_ro[2] / (-2 * K_A_ro[0])
        y0gA = K_A_ro[3] / (-2 * K_A_ro[1])
        x0gB = K_B_ro[2] / (-2 * K_B_ro[0])
        y0gB = K_B_ro[3] / (-2 * K_B_ro[1])
        distance = math.sqrt((x0gA - x0gB) ** 2 + (y0gA - y0gB) ** 2)


        result_list = [distance, 1 / (-2 * K_A_ro[0]), 1 / (-2 * K_A_ro[1]), 1 / (-2 * K_B_ro[0]), 1 / (-2 * K_B_ro[1])]
        print("result:", result_list)
        dataframe = dataframe.append(pd.Series(result_list, ['distance', 'sigma_x_A', 'sigma_y_A', 'sigma_x_B', 'sigma_y_B']), ignore_index=True)

    dataframe.to_csv('result_cnn.csv')
