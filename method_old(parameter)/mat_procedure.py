import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy.io as scio
from sklearn.cluster import KMeans
from sklearn import mixture
import random
import math
import cv2


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
    return K


def Gaussian_fit_rotation(x, y, value):
    best_score = 999999999
    best_angle = None
    # K = np.array([])
    for i in range(0, 100):
        # 创建随机解
        angle = random.uniform(0, 90)
        radian = angle * np.pi / 180

        x_new = np.array(x) * np.cos(radian) - np.array(y) * np.sin(radian)
        y_new = np.array(x) * np.sin(radian) + np.array(y) * np.cos(radian)
        M = Gaussian_fit(x_new, y_new, np.array(value))

        # 计算成本值
        current_score = (np.sqrt(1 / (-2 * M[0])) + np.sqrt(1 / (-2 * M[1]))) / abs(np.sqrt(1 / (-2 * M[0])) - np.sqrt(1 / (-2 * M[1])))

        # 与目前得到的最优解进行比较
        if current_score < best_score:
            best_score = current_score
            best_angle = angle
            K = M

    K = np.append(K, best_angle)

    return K


dataframe = pd.DataFrame(columns=['scale', 'x_A', 'x_B', 'y_A', 'y_B', 'sigma_x_A', 'sigma_x_B', 'sigma_y_A', 'sigma_y_B'])

comparisonFile = pd.DataFrame(columns=['cnn', 'mat_scale', 'coefficient'])

initData = pd.read_pickle('matData.pkl')

df = pd.read_csv('result_cnn_7.csv')


for index, row in df.iterrows():
    factor_cnn = (row['sigma_x_A'] + row['sigma_x_B'] + row['sigma_y_A'] + row['sigma_y_B']) / math.sqrt((row['x_A'] - row['x_B']) ** 2 + (row['y_A'] - row['y_B']) ** 2)

    factor_scale_mat = 0

    best_score = 999

    result_list_submit = []

    print("-------------------------")
    for i in range(3):

        # 进行多次随机比例的裁剪
        k = random.randint(0, 150)
        averageData = initData[k:600-k, k:600-k]
        scale = k / 150
        # print("k, scale = ", k, scale)

        # 将裁剪后的图像缩放至标准大小
        cell_used_single = cv2.resize(averageData, (128, 128), interpolation=cv2.INTER_AREA)
        cell_used_single = np.uint8(np.clip(cell_used_single, 0, 1) * 255.0)
        # plt.figure()
        # plt.subplot(331)
        # plt.imshow(cell_used_single)


        clst = mixture.GaussianMixture(n_components=3, max_iter=100, covariance_type="full")
        clst.fit(cell_used_single.reshape(-1, 1))
        predicted_labels = clst.predict(cell_used_single.reshape(-1, 1))
        centroids = clst.means_
        labels = predicted_labels
        # print("centroids", centroids)
        # print("length of labels", len(labels))
        label_matrix = labels.reshape((128, 128))
        # print("label_matrix", label_matrix)

        # plt.subplot(332)
        # plt.imshow(label_matrix)


        # 找到需要用的那两类
        min_index = np.argmin(centroids)
        max_index = np.argmax(centroids)
        # print("min_index = ", min_index)
        # print("max_index = ", max_index)

        # drop_index = list(filter(lambda x: (x != np.argmin(centroids)) & (x != np.argmax(centroids)), [0, 1, 2]))[0]
        # print("drop_index = ", drop_index)

        data_gmm_0 = cell_used_single.copy()
        for i in range(128):
            for j in range(128):
                if label_matrix[i, j] == min_index:
                    data_gmm_0[i][j] = 1
                else:
                    data_gmm_0[i][j] = 0
        # plt.subplot(334)
        # plt.axis('off')
        data_0 = np.uint8(np.clip(data_gmm_0, 0, 1) * 255.0)
        # plt.imshow(data_0)

        data_gmm_1 = cell_used_single.copy()
        for i in range(128):
            for j in range(128):
                if label_matrix[i, j] == max_index:
                    data_gmm_1[i][j] = 1
                else:
                    data_gmm_1[i][j] = 0
        # plt.subplot(335)
        # plt.axis('off')
        data_1 = np.uint8(np.clip(data_gmm_1, 0, 1) * 255.0)
        # plt.imshow(data_1)


        data_A = data_0
        data_B = data_1
        class_A = min_index
        class_B = max_index
        avg_A = centroids[min_index]
        avg_B = centroids[max_index]


        x_A, y_A, value_A, x_B, y_B, value_B = ([] for _ in range(6))
        for i in range(128):
            for j in range(128):
                # if cell_used_single[i, j] != 0:
                if label_matrix[i, j] == class_A:
                    x_A.append(i)
                    y_A.append(j)
                    value_A.append(cell_used_single[i, j] * 1000000)
                    # value_A.append(1)
                else:
                    x_A.append(i)
                    y_A.append(j)
                    value_A.append(0.01)
                if label_matrix[i, j] == class_B:
                    x_B.append(i)
                    y_B.append(j)
                    value_B.append(cell_used_single[i, j] * 1000000)
                    # value_B.append(1)
                else:
                    x_B.append(i)
                    y_B.append(j)
                    value_B.append(0.01)

        # print("x_A: ", x_A)
        # print("y_A: ", y_A)
        # print("value_A: ", value_A)
        # print("length of x_A, y_A, value_A", len(x_A), len(y_A), len(value_A))
        # print("x_B: ", x_B)
        # print("y_B: ", y_B)
        # print("value_B: ", value_B)
        # print("length of x_B, y_B, value_B", len(x_B), len(y_B), len(value_B))

        # print("-------------------------------------------------------------")

        try:
            K_A_ro = Gaussian_fit_rotation(np.array(x_A), np.array(y_A), np.array(value_A))
            K_B_ro = Gaussian_fit_rotation(np.array(x_B), np.array(y_B), np.array(value_B))
        except UnboundLocalError:
            print("Wrong!Wrong!Wrong!")
            continue

        # print("K_A_ro = ", K_A_ro)
        # print("K_B_ro = ", K_B_ro)
        # print("x0_gaussian_A_ro, y0_gaussian_A_ro = ", K_A_ro[2] / (-2 * K_A_ro[0]), K_A_ro[3] / (-2 * K_A_ro[1]))
        # print("x0_gaussian_B_ro, y0_gaussian_B_ro = ", K_B_ro[2] / (-2 * K_B_ro[0]), K_B_ro[3] / (-2 * K_B_ro[1]))
        # print("sigma_x²_gaussian_A_ro, sigma_y²_gaussian_A_ro = ", 1 / (-2 * K_A_ro[0]), 1 / (-2 * K_A_ro[1]))
        # print("sigma_x²_gaussian_B_ro, sigma_y²_gaussian_B_ro = ", 1 / (-2 * K_B_ro[0]), 1 / (-2 * K_B_ro[1]))
        # print("rotation: ", K_A_ro[5])
        # print("rotation: ", K_B_ro[5])

        x0gA = K_A_ro[2] / (-2 * K_A_ro[0])
        y0gA = K_A_ro[3] / (-2 * K_A_ro[1])
        x0gB = K_B_ro[2] / (-2 * K_B_ro[0])
        y0gB = K_B_ro[3] / (-2 * K_B_ro[1])

        x_return_A = np.array(x0gA) * np.cos(K_A_ro[5] * np.pi / 180) + np.array(y0gA) * np.sin(K_A_ro[5] * np.pi / 180)
        y_return_A = np.array(y0gA) * np.cos(K_A_ro[5] * np.pi / 180) - np.array(x0gA) * np.sin(K_A_ro[5] * np.pi / 180)
        x_return_B = np.array(x0gB) * np.cos(K_B_ro[5] * np.pi / 180) + np.array(y0gB) * np.sin(K_B_ro[5] * np.pi / 180)
        y_return_B = np.array(y0gB) * np.cos(K_B_ro[5] * np.pi / 180) - np.array(x0gB) * np.sin(K_B_ro[5] * np.pi / 180)

        # print("x_return_A, y_return_A = ", x_return_A, y_return_A)
        # print("x_return_B, y_return_B = ", x_return_B, y_return_B)

        # scale = 1
        sigma_x_A = np.sqrt(1 / (-2 * K_A_ro[0]))
        sigma_x_B = np.sqrt(1 / (-2 * K_B_ro[0]))
        sigma_y_A = np.sqrt(1 / (-2 * K_A_ro[1]))
        sigma_y_B = np.sqrt(1 / (-2 * K_B_ro[1]))
        # print("sigma_x_gaussian_A, sigma_y_gaussian_A = ", sigma_x_A, sigma_y_A)
        # print("sigma_x_gaussian_B, sigma_y_gaussian_B = ", sigma_x_B, sigma_y_B)


        result_list = [scale, x_return_A, x_return_B, y_return_A, y_return_B, sigma_x_A, sigma_x_B, sigma_y_A, sigma_y_B]
        # print("result:", result_list)


        factor_scale_mat_current = (result_list[1] + result_list[2] + result_list[3] + result_list[4]) / math.sqrt((result_list[5] - result_list[6]) ** 2 + (result_list[7] - result_list[8]) ** 2)

        current_score = abs(factor_scale_mat_current - factor_cnn)
        print("factor_scale_mat_current = {}, factor_cnn = {}".format(factor_scale_mat_current, factor_cnn))
        if current_score < best_score:
            best_score = current_score
            factor_scale_mat = factor_scale_mat_current
            result_list_submit = result_list

    # 最后需要用这个factor定义个范围来做判断
    print("factor_cnn VS factor_scale_mat  ----  ", factor_cnn, factor_scale_mat)
    coefficient = abs(factor_cnn - factor_scale_mat) / factor_cnn

    comparisonList = [factor_cnn, factor_scale_mat, coefficient]
    comparisonFile = comparisonFile.append(pd.Series(comparisonList, ['cnn', 'mat_scale', 'coefficient']), ignore_index=True)

    dataframe = dataframe.append(pd.Series(result_list_submit, ['scale', 'x_A', 'x_B', 'y_A', 'y_B', 'sigma_x_A', 'sigma_x_B', 'sigma_y_A', 'sigma_y_B']), ignore_index=True)

dataframe.to_csv('result_mat_all.csv')
comparisonFile.to_csv('comparison_all.csv')

# plt.show()