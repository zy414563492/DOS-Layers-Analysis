import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import cv2
import scipy.optimize
from sklearn import mixture
from sklearn.cluster import KMeans


def make_gaussian(g_size, p):
    gaussian = np.zeros((g_size, g_size))

    rotation_mat = np.array([[np.cos(p[5]), -np.sin(p[5])], [np.sin(p[5]), np.cos(p[5])]])
    center_vert = np.array([p[1], p[2]]).reshape((-1, 1))

    for x in range(g_size):
        for y in range(g_size):
            point = np.array([x, y]).reshape((-1, 1))
            transrated_point = np.dot(rotation_mat, (point - center_vert))

            gaussian[x, y] = p[0] * np.exp((-1 / 2) * ((transrated_point[0] / p[3]) ** 2 + (transrated_point[1] / p[4]) ** 2))

    return gaussian


def make_gaussians2_sum(num_gaus, g_size, p):
    gaussian = np.zeros((g_size, g_size))

    for n in range(num_gaus):
        gaussian += make_gaussian(g_size, p[n * 6: (n + 1) * 6])

    return gaussian


def fine_tuning(g_size, p, scale, theta_all, trans_x, trans_y):
    k = p.copy()
    x_mid = (k[1] + k[7]) / 2
    y_mid = (k[2] + k[8]) / 2
    x1_new = np.array(k[1] - x_mid) * np.cos(theta_all) - np.array(k[2] - y_mid) * np.sin(theta_all) + x_mid
    y1_new = np.array(k[1] - x_mid) * np.sin(theta_all) + np.array(k[2] - y_mid) * np.cos(theta_all) + y_mid
    x2_new = np.array(k[7] - x_mid) * np.cos(theta_all) - np.array(k[8] - y_mid) * np.sin(theta_all) + x_mid
    y2_new = np.array(k[7] - x_mid) * np.sin(theta_all) + np.array(k[8] - y_mid) * np.cos(theta_all) + y_mid
    # x1_new = np.array(k[1] - 64) * np.cos(theta_all) - np.array(k[2] - 64) * np.sin(theta_all) + 64
    # y1_new = np.array(k[1] - 64) * np.sin(theta_all) + np.array(k[2] - 64) * np.cos(theta_all) + 64
    # x2_new = np.array(k[7] - 64) * np.cos(theta_all) - np.array(k[8] - 64) * np.sin(theta_all) + 64
    # y2_new = np.array(k[7] - 64) * np.sin(theta_all) + np.array(k[8] - 64) * np.cos(theta_all) + 64
    k[1] = x1_new
    k[2] = y1_new
    k[7] = x2_new
    k[8] = y2_new
    k[1] += g_size * (scale - 1) / 2
    k[2] += g_size * (scale - 1) / 2
    k[7] += g_size * (scale - 1) / 2
    k[8] += g_size * (scale - 1) / 2
    k[1] += trans_x * scale
    k[2] += trans_y * scale
    k[7] += trans_x * scale
    k[8] += trans_y * scale
    k[5] -= theta_all
    k[11] -= theta_all
    return k


def make_tuned_gaussians(g_size, p, x):
    if x[0] != 1.65:
        return np.ones((g_size, g_size))
    param_new = fine_tuning(g_size, p, x[0], x[1], x[2], x[3])
    # print("x:", x)
    new_size = int(g_size * x[0])
    tuned_img = make_gaussians2_sum(2, new_size, param_new)
    tuned_gaussian = cv2.resize(tuned_img, (g_size, g_size), interpolation=cv2.INTER_AREA)
    return tuned_gaussian



dataframe = pd.DataFrame(columns=['score', 'scale', 'theta_all', 'trans_x', 'trans_y', 'pro1', 'pro2'])

tgt_layer = 'convolution_7'
# units = [0, 16, 26, 33, 37, 38, 47, 97]
# u_num = len(units)
# unit_use = 6

units = list(range(0, 512))
u_num = len(units)

data_all = np.zeros((u_num, 128, 128, 3))
data_black = np.zeros((128, 128, 3))
for i in range(u_num):
    data_all[i] = pd.read_pickle("./visualize_laplacian/7-15/visualize-lap-{}-{}-15.pkl".format(tgt_layer, units[i]))[64:192, 64:192]

for unit_use in range(0, 20):
    print("unit_use", unit_use)
    data_row = data_all[unit_use]
    data_row = np.clip(data_row, 0., 1.)  # 试下不加的效果

    ##################### ↓↓↓ CNN图像预处理 ↓↓↓ #####################

    # 先高斯模糊去掉部分噪点
    data_blur = cv2.GaussianBlur(data_row, (3, 3), 0)

    # 先转化为二维数据
    image_array = data_blur.reshape((data_blur.shape[0] * data_blur.shape[1], 3))

    # 1st: GMM
    clst = mixture.GaussianMixture(n_components=3, max_iter=100, covariance_type="full")
    clst.fit(image_array)
    predicted_labels = clst.predict(image_array)
    centroids_GMM = clst.means_
    print("centroids_GMM", centroids_GMM)
    labels = predicted_labels
    label_matrix = labels.reshape((128, 128))

    # 找中心的那一块， 求出最大值的索引
    baseline = np.array([0.5] * 9).reshape(3, 3)

    max_index = np.argmax(np.sum(abs(centroids_GMM - baseline), axis=1))
    center_mean = centroids_GMM[max_index].dot(np.array([0.2989, 0.5870, 0.1140]).T)
    print("center_mean:", center_mean)

    # 得到灰度图，但如果直接用的话效果不好，仍需在此图上进行进一步调整
    data_filled = data_blur.dot(np.array([0.2989, 0.5870, 0.1140]).T) - center_mean

    # 从原始图像中将除中心区域外的区域变成中心平均值
    data_center = data_blur.copy()
    if max_index == 0:
        for i in range(128):
            for j in range(128):
                if label_matrix[i, j] != 0:
                    data_center[i][j] = [center_mean, center_mean,
                                         center_mean]  # [0.5, 0.5, 0.5]  # [0, 0, 0]  # [np.nan, np.nan, np.nan] # 四周平均

    elif max_index == 1:
        for i in range(128):
            for j in range(128):
                if label_matrix[i, j] != 1:
                    data_center[i][j] = [center_mean, center_mean, center_mean]
    else:
        for i in range(128):
            for j in range(128):
                if label_matrix[i, j] != 2:
                    data_center[i][j] = [center_mean, center_mean, center_mean]

    data_filled_center = data_center.dot(np.array([0.2989, 0.5870, 0.1140]).T) - center_mean

    # 处理后的图像再次展开为二维数据
    image_array_new = data_center.reshape((data_row.shape[0] * data_row.shape[1], 3))

    # 2nd: k-means聚类
    kmeans = KMeans(n_clusters=3, init='k-means++')
    kmeans.fit(image_array_new)
    centroids_kmeans = kmeans.cluster_centers_
    labels = kmeans.labels_
    print("centroids_k-means", centroids_kmeans)
    label_matrix_new = labels.reshape((128, 128))

    # 上步在处理过后的中心数据集上进行k-means后分成了三类，这里分别黑白显示
    data_0_new = data_center.copy()
    data_1_new = data_center.copy()
    data_2_new = data_center.copy()
    for i in range(128):
        for j in range(128):
            if label_matrix_new[i, j] == 0:
                data_0_new[i][j] = [1, 1, 1]
            else:
                data_0_new[i][j] = [0, 0, 0]
            if label_matrix_new[i, j] == 1:
                data_1_new[i][j] = [1, 1, 1]
            else:
                data_1_new[i][j] = [0, 0, 0]
            if label_matrix_new[i, j] == 2:
                data_2_new[i][j] = [1, 1, 1]
            else:
                data_2_new[i][j] = [0, 0, 0]

    # 确定促进和抑制类
    centroids_filled = centroids_kmeans.dot(np.array([0.2989, 0.5870, 0.1140]).T)
    pos_index = np.argmax(centroids_filled)
    neg_index = np.argmin(centroids_filled)
    print("pos_index, neg_index = ", pos_index, neg_index)
    data_pos = data_0_new if pos_index == 0 else (data_1_new if pos_index == 1 else data_2_new)
    data_neg = data_0_new if neg_index == 0 else (data_1_new if neg_index == 1 else data_2_new)

    # 膨胀与腐蚀，去掉离散点
    kernel = np.ones((4, 4), np.uint8)
    data_pos = cv2.morphologyEx(cv2.morphologyEx(data_pos, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
    data_neg = cv2.morphologyEx(cv2.morphologyEx(data_neg, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)

    data_mod = data_filled_center.copy()
    for i in range(128):
        for j in range(128):
            if (data_pos[i, j] == [1, 1, 1]).all() and (data_neg[i, j] != [1, 1, 1]).all():
                data_mod[i][j] = abs(data_mod[i][j])
            elif (data_neg[i, j] == [1, 1, 1]).all() and (data_pos[i, j] != [1, 1, 1]).all():
                data_mod[i][j] = -abs(data_mod[i][j])
            elif (data_pos[i, j] == [1, 1, 1]).all() and (data_neg[i, j] == [1, 1, 1]).all():
                data_mod[i][j] = data_mod[i][j] / 2

    # 最后在平均模糊一下
    data_mod = cv2.blur(data_mod, (2, 2))

    score_mod = np.sqrt(np.sum(data_mod ** 2))
    print("score_mod = ", score_mod)

    ##################### ↑↑↑ CNN图像预处理 ↑↑↑ #####################

    n_pos = 1
    n_neg = 1

    stim_size = 128
    param_init = [0.4356, 78.61, 64.08, 19.72, 12.39, 1.477,
                  -0.38804, 48.145, 64.923, 32, 32, 0.089]

    # 约束设置
    # min_scale = 1.5
    # max_scale = 3.5
    min_theta = -np.pi
    max_theta = np.pi
    # min_trans = -int(stim_size / 6)
    # max_trans = int(stim_size / 6)
    x_trans = -8
    y_trans = -1


    objfcn = lambda x: np.sqrt(np.sum((data_mod - make_tuned_gaussians(stim_size, param_init, x)) ** 2))


    # 开始fitting
    score = np.inf
    param = np.zeros(4)
    max_itr = 5
    for itr in range(max_itr * (n_pos + n_neg)):
        print("itr:", itr)

        # 随机设置初始值
        init_x = np.array([1.65,  # (max_scale - min_scale) * random.random() + min_scale,
                           (max_theta - min_theta) * random.random() + min_theta,
                           x_trans,  # (max_trans - min_trans) * random.random() + min_trans,
                           y_trans])  # (max_trans - min_trans) * random.random() + min_trans])

        # init_x = np.array([1.8, 3.01, 3.01, 0, -13])

        # SLSQP约束方法
        cons = ({'type': 'eq', 'fun': lambda x: x[0] - 1.65},  # {'type': 'ineq', 'fun': lambda x: x[0] - min_scale}, {'type': 'ineq', 'fun': lambda x: -x[0] + max_scale},
                {'type': 'ineq', 'fun': lambda x: x[1] - min_theta}, {'type': 'ineq', 'fun': lambda x: -x[1] + max_theta},
                {'type': 'eq', 'fun': lambda x: x[2] - x_trans},   # {'type': 'ineq', 'fun': lambda x: x[2] - min_trans}, {'type': 'ineq', 'fun': lambda x: -x[2] + max_trans},
                {'type': 'eq', 'fun': lambda x: x[3] - y_trans})  # {'type': 'ineq', 'fun': lambda x: x[3] - min_trans}, {'type': 'ineq', 'fun': lambda x: -x[3] + max_trans})

        res = scipy.optimize.minimize(objfcn, init_x, method='SLSQP', constraints=cons)

        cur_success = res.success
        cur_score = res.fun
        cur_param = res.x
        print("cur_success:", res.success)
        print("cur_score:", cur_score)
        print("cur_param:", cur_param)

        if cur_score < score and cur_success == True:
            score = cur_score
            param = cur_param

    print("score:", score)
    print("param:", param)

    score_gaussian = np.sqrt(np.sum(make_tuned_gaussians(stim_size, param_init, param) ** 2))
    print("score_gaussian:", score_gaussian)
    pro1 = score / score_gaussian
    pro2 = score / score_mod
    print("pro1 = {}, pro2 = {}".format(pro1, pro2))


    result_list = [score, param[0], param[1], param[2], param[3], pro1, pro2]
    # print("result:", result_list)
    dataframe = dataframe.append(pd.Series(result_list, ['score', 'scale', 'theta_all', 'trans_x', 'trans_y', 'pro1', 'pro2']), ignore_index=True)

dataframe.to_csv('tuning_param_all.csv')








