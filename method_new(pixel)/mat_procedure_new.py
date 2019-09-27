import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# import scipy.io as scio
# from sklearn.cluster import KMeans
# from sklearn import mixture
import random
# import math
import cv2
import scipy.optimize


# def Gaussian_fit(x, y, value):
#     C = np.array([np.sum(x ** 2 * np.log(value)),
#                   np.sum(y ** 2 * np.log(value)),
#                   np.sum(x * np.log(value)),
#                   np.sum(y * np.log(value)),
#                   np.sum(np.log(value))])
#
#     B = np.array([[np.sum(x ** 4), np.sum(x ** 2 * y ** 2), np.sum(x ** 3), np.sum(x ** 2 * y), np.sum(x ** 2)],
#                   [np.sum(x ** 2 * y ** 2), np.sum(y ** 4), np.sum(x * y ** 2), np.sum(y ** 3), np.sum(y ** 2)],
#                   [np.sum(x ** 3), np.sum(x * y ** 2), np.sum(x ** 2), np.sum(x * y), np.sum(x)],
#                   [np.sum(x ** 2 * y), np.sum(y ** 3), np.sum(x * y), np.sum(y ** 2), np.sum(y)],
#                   [np.sum(x ** 2), np.sum(y ** 2), np.sum(x), np.sum(y), len(value)]])
#
#     B_inv = np.linalg.inv(B)
#     K = B_inv.dot(C.T)
#     return K
#
#
# def Gaussian_fit_rotation(x, y, value):
#     best_score = 999999999
#     best_angle = None
#     for i in range(0, 100):
#         # 创建随机解
#         angle = random.uniform(0, 90)
#         radian = angle * np.pi / 180
#
#         x_new = np.array(x) * np.cos(radian) - np.array(y) * np.sin(radian)
#         y_new = np.array(x) * np.sin(radian) + np.array(y) * np.cos(radian)
#         M = Gaussian_fit(x_new, y_new, np.array(value))
#
#         # 计算成本值
#         current_score = (np.sqrt(1 / (-2 * M[0])) + np.sqrt(1 / (-2 * M[1]))) / abs(np.sqrt(1 / (-2 * M[0])) - np.sqrt(1 / (-2 * M[1])))
#
#         # 与目前得到的最优解进行比较
#         if current_score < best_score:
#             best_score = current_score
#             best_angle = angle
#             K = M
#
#     K = np.append(K, best_angle)
#
#     return K

# averageData = pd.read_pickle('matData.pkl')
# print(averageData)

# k = random.randint(0, 150)
# averageData = averageData[k:600-k, k:600-k]
# scale = k / 150
# print("k, scale = ", k, scale)


# cell_used_single = cv2.resize(averageData, (128, 128), interpolation=cv2.INTER_AREA)
# cell_used_single = np.uint8(np.clip(cell_used_single, 0, 1) * 255.0)
# plt.figure()
# plt.subplot(331)
# plt.imshow(cell_used_single)
#
#
# clst = mixture.GaussianMixture(n_components=3, max_iter=100, covariance_type="full")
# clst.fit(cell_used_single.reshape(-1, 1))
# predicted_labels = clst.predict(cell_used_single.reshape(-1, 1))
# centroids = clst.means_
# labels = predicted_labels
# print("centroids", centroids)
# print("length of labels", len(labels))
# label_matrix = labels.reshape((128, 128))
# print("label_matrix", label_matrix)
#
# plt.subplot(332)
# plt.imshow(label_matrix)
#
#
# # 找到需要用的那两类
# min_index = np.argmin(centroids)
# max_index = np.argmax(centroids)
# print("min_index = ", min_index)
# print("max_index = ", max_index)
#
# # drop_index = list(filter(lambda x: (x != np.argmin(centroids)) & (x != np.argmax(centroids)), [0, 1, 2]))[0]
# # print("drop_index = ", drop_index)
#
# data_gmm_0 = cell_used_single.copy()
# for i in range(128):
#     for j in range(128):
#         if label_matrix[i, j] == min_index:
#             data_gmm_0[i][j] = 1
#         else:
#             data_gmm_0[i][j] = 0
# plt.subplot(334)
# plt.axis('off')
# data_0 = np.uint8(np.clip(data_gmm_0, 0, 1) * 255.0)
# plt.imshow(data_0)
#
# data_gmm_1 = cell_used_single.copy()
# for i in range(128):
#     for j in range(128):
#         if label_matrix[i, j] == max_index:
#             data_gmm_1[i][j] = 1
#         else:
#             data_gmm_1[i][j] = 0
# plt.subplot(335)
# plt.axis('off')
# data_1 = np.uint8(np.clip(data_gmm_1, 0, 1) * 255.0)
# plt.imshow(data_1)
#
#
# data_A = data_0
# data_B = data_1
# class_A = min_index
# class_B = max_index
# avg_A = centroids[min_index]
# avg_B = centroids[max_index]
#
#
# x_A, y_A, value_A, x_B, y_B, value_B = ([] for _ in range(6))
# for i in range(128):
#     for j in range(128):
#         # if cell_used_single[i, j] != 0:
#         if label_matrix[i, j] == class_A:
#             x_A.append(i)
#             y_A.append(j)
#             value_A.append(cell_used_single[i, j] * 1000000)
#             # value_A.append(1)
#         else:
#             x_A.append(i)
#             y_A.append(j)
#             value_A.append(0.01)
#         if label_matrix[i, j] == class_B:
#             x_B.append(i)
#             y_B.append(j)
#             value_B.append(cell_used_single[i, j] * 1000000)
#             # value_B.append(1)
#         else:
#             x_B.append(i)
#             y_B.append(j)
#             value_B.append(0.01)
#
# print("x_A: ", x_A)
# print("y_A: ", y_A)
# print("value_A: ", value_A)
# print("length of x_A, y_A, value_A", len(x_A), len(y_A), len(value_A))
# print("x_B: ", x_B)
# print("y_B: ", y_B)
# print("value_B: ", value_B)
# print("length of x_B, y_B, value_B", len(x_B), len(y_B), len(value_B))
#
# print("-------------------------------------------------------------")
#
# K_A_ro = Gaussian_fit_rotation(np.array(x_A), np.array(y_A), np.array(value_A))
# K_B_ro = Gaussian_fit_rotation(np.array(x_B), np.array(y_B), np.array(value_B))
# print("K_A_ro = ", K_A_ro)
# print("K_B_ro = ", K_B_ro)
# print("x0_gaussian_A_ro, y0_gaussian_A_ro = ", K_A_ro[2] / (-2 * K_A_ro[0]), K_A_ro[3] / (-2 * K_A_ro[1]))
# print("x0_gaussian_B_ro, y0_gaussian_B_ro = ", K_B_ro[2] / (-2 * K_B_ro[0]), K_B_ro[3] / (-2 * K_B_ro[1]))
# print("sigma_x²_gaussian_A_ro, sigma_y²_gaussian_A_ro = ", 1 / (-2 * K_A_ro[0]), 1 / (-2 * K_A_ro[1]))
# print("sigma_x²_gaussian_B_ro, sigma_y²_gaussian_B_ro = ", 1 / (-2 * K_B_ro[0]), 1 / (-2 * K_B_ro[1]))
# print("rotation: ", K_A_ro[5])
# print("rotation: ", K_B_ro[5])
#
# x0gA = K_A_ro[2] / (-2 * K_A_ro[0])
# y0gA = K_A_ro[3] / (-2 * K_A_ro[1])
# x0gB = K_B_ro[2] / (-2 * K_B_ro[0])
# y0gB = K_B_ro[3] / (-2 * K_B_ro[1])
#
# x_return_A = np.array(x0gA) * np.cos(K_A_ro[5] * np.pi / 180) + np.array(y0gA) * np.sin(K_A_ro[5] * np.pi / 180)
# y_return_A = np.array(y0gA) * np.cos(K_A_ro[5] * np.pi / 180) - np.array(x0gA) * np.sin(K_A_ro[5] * np.pi / 180)
# x_return_B = np.array(x0gB) * np.cos(K_B_ro[5] * np.pi / 180) + np.array(y0gB) * np.sin(K_B_ro[5] * np.pi / 180)
# y_return_B = np.array(y0gB) * np.cos(K_B_ro[5] * np.pi / 180) - np.array(x0gB) * np.sin(K_B_ro[5] * np.pi / 180)
#
# print("x_return_A, y_return_A = ", x_return_A, y_return_A)
# print("x_return_B, y_return_B = ", x_return_B, y_return_B)
#
# # scale = 1
# sigma_x_A = np.sqrt(1 / (-2 * K_A_ro[0]))
# sigma_x_B = np.sqrt(1 / (-2 * K_B_ro[0]))
# sigma_y_A = np.sqrt(1 / (-2 * K_A_ro[1]))
# sigma_y_B = np.sqrt(1 / (-2 * K_B_ro[1]))
# print("sigma_x_gaussian_A, sigma_y_gaussian_A = ", sigma_x_A, sigma_y_A)
# print("sigma_x_gaussian_B, sigma_y_gaussian_B = ", sigma_x_B, sigma_y_B)
#
#
#
# plt.show()







def make_gaussian(g_size, p):
    gaussian = np.zeros((g_size, g_size))

    rotation_mat = np.array([[np.cos(p[5]), -np.sin(p[5])], [np.sin(p[5]), np.cos(p[5])]])
    center_vert = np.array([p[1], p[2]]).reshape((-1, 1))

    for x in range(g_size):
        for y in range(g_size):
            point = np.array([x, y]).reshape((-1, 1))
            transrated_point = np.dot(rotation_mat, (point - center_vert))

            # 这里查看p很有用！
            # print("make_gaussian_p:", p)

            gaussian[x, y] = p[0] * np.exp((-1 / 2) * ((transrated_point[0] / p[3]) ** 2 + (transrated_point[1] / p[4]) ** 2))

    return gaussian


def make_gaussians2_sum(num_gaus, g_size, p):
    gaussian = np.zeros((g_size, g_size))

    for n in range(num_gaus):
        gaussian += make_gaussian(g_size, p[n * 6: (n + 1) * 6])

    return gaussian


def gaussian_fitting(n_pos, n_neg, img):

    stim_size = img.shape[0]
    score = np.inf

    objfcn = lambda x: np.sum((img - make_gaussians2_sum(n_pos+n_neg, stim_size, x)) ** 2)

    # gaussian parameters
    min_scale = 0
    max_scale = max(abs(np.min(img)), abs(np.max(img)))
    min_center = 1
    max_center = stim_size
    min_sigma = 1
    max_sigma = int(stim_size / 2)
    min_theta = 0
    max_theta = np.pi / 2
    print("max_scale: ", max_scale)

    param = np.zeros(12)
    max_itr = 1
    for itr in range(max_itr * (n_pos + n_neg)):
        print("itr:", itr)

        # 随机设置初始值
        init_param_pos = np.array([max_scale * random.random(),
                                   (max_center - min_center) * random.random() + min_center,
                                   (max_center - min_center) * random.random() + min_center,
                                   (max_sigma - min_sigma) * random.random() + min_sigma,
                                   (max_sigma - min_sigma) * random.random() + min_sigma,
                                   (max_theta - min_theta) * random.random() + min_theta])

        init_param_neg = np.array([-max_scale * random.random(),
                                   (max_center - min_center) * random.random() + min_center,
                                   (max_center - min_center) * random.random() + min_center,
                                   (max_sigma - min_sigma) * random.random() + min_sigma,
                                   (max_sigma - min_sigma) * random.random() + min_sigma,
                                   (max_theta - min_theta) * random.random() + min_theta])

        init_param = np.hstack((init_param_pos, init_param_neg))


        # SLSQP约束方法
        cons = ({'type': 'ineq', 'fun': lambda x: x[0] - min_scale}, {'type': 'ineq', 'fun': lambda x: -x[0] + max_scale},
                {'type': 'ineq', 'fun': lambda x: x[1] - min_center}, {'type': 'ineq', 'fun': lambda x: -x[1] + max_center},
                {'type': 'ineq', 'fun': lambda x: x[2] - min_center}, {'type': 'ineq', 'fun': lambda x: -x[2] + max_center},
                {'type': 'ineq', 'fun': lambda x: x[3] - min_sigma}, {'type': 'ineq', 'fun': lambda x: -x[3] + max_sigma},
                {'type': 'ineq', 'fun': lambda x: x[4] - min_sigma}, {'type': 'ineq', 'fun': lambda x: -x[4] + max_sigma},
                {'type': 'ineq', 'fun': lambda x: x[5] - min_theta}, {'type': 'ineq', 'fun': lambda x: -x[5] + max_theta},
                {'type': 'ineq', 'fun': lambda x: -x[6] - min_scale}, {'type': 'ineq', 'fun': lambda x: x[6] + max_scale},
                {'type': 'ineq', 'fun': lambda x: x[7] - min_center}, {'type': 'ineq', 'fun': lambda x: -x[7] + max_center},
                {'type': 'ineq', 'fun': lambda x: x[8] - min_center}, {'type': 'ineq', 'fun': lambda x: -x[8] + max_center},
                {'type': 'ineq', 'fun': lambda x: x[9] - min_sigma}, {'type': 'ineq', 'fun': lambda x: -x[9] + max_sigma},
                {'type': 'ineq', 'fun': lambda x: x[10] - min_sigma}, {'type': 'ineq', 'fun': lambda x: -x[10] + max_sigma},
                {'type': 'ineq', 'fun': lambda x: x[11] - min_theta}, {'type': 'ineq', 'fun': lambda x: -x[11] + max_theta})

        res = scipy.optimize.minimize(objfcn, init_param, method='SLSQP', constraints=cons)


        # L-BFGS-B边界方法（效果差）
        # bnd = [(min_scale, max_scale), (min_center, max_center), (min_center, max_center), (min_sigma, max_sigma), (min_sigma, max_sigma), (min_theta, max_theta), (-max_scale, min_scale), (min_center, max_center), (min_center, max_center), (min_sigma, max_sigma), (max_sigma, max_sigma), (min_theta, max_theta)]
        # print("bnd:", bnd)
        # res = scipy.optimize.minimize(objfcn, init_param, method='L-BFGS-B', bounds=bnd)


        cur_score = res.fun
        cur_param = res.x

        if cur_score < score:
            score = cur_score
            param = cur_param

        print("cur_success:", res.success)
        print("cur_score:", cur_score)
        # print("cur_param:", cur_param)

    return param, score


n_pos = 1
n_neg = 1

img_init = pd.read_pickle('matData.pkl')
img = img_init[150:450, 150:450]
img -= 0.5
img_used = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

plt.figure()
plt.imshow(img_used)
plt.show()


# a = np.array([[0, 0, 1, 0, 0],
#               [1, 2, 3, 2, 1],
#               [0, 0, 1, 0, 0],
#               [0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0]])
# b = np.array([[0, 0, 0, 0, 0],
#               [0, 0, 1, 0, 0],
#               [0, 1, 2, 1, 0],
#               [1, 3, 4, 3, 1],
#               [0, 1, 2, 1, 0]])
#
# c = np.array([[0, 0, 1, 0, 0],
#               [1, 2, 3, 2, 1],
#               [0, 0, 1, -1, 0],
#               [0, 0, -1, -4, -1],
#               [0, 0, 0, -1, 0]])
#
# img_used = (a - b) / 10
# print(img_used)

# print(img_used.shape, type(img_used), img_used)
#
# param, score = gaussian_fitting(n_pos, n_neg, img_used)
#
# print("score:", score)
# print("param:", param)



print("----------------")



#
#
# def fun(args):
#     a = args
#     v = lambda x: x[0] + a * x[1]
#     return v
#
#
# def con(args):
#     x0min, x0max, x1min, x1max = args
#     cons = ({'type': 'ineq', 'fun': lambda x: x[0] - x0min},
#             {'type': 'ineq', 'fun': lambda x: -x[0] + x0max},
#             {'type': 'ineq', 'fun': lambda x: x[1] - x1min},
#             {'type': 'ineq', 'fun': lambda x: -x[1] + x1max},
#             {'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 3})
#     return cons
#
# args = (2)
# args1 = (1, 5, 1, 5)
# cons = con(args1)
#
# x0 = np.asarray((10, 100))
# res = scipy.optimize.minimize(fun(args), x0, method='SLSQP', constraints=cons)
# print(res.fun)
# print(res.success)
# print(res.x)




# p = np.array([[1, 2, 3, 4, 5, 6], [-6, 5, 4, 3, 2, 1]])
# p = np.array([1, 2, 3, 4, 5, 6])
# p0min = 0.5
# result = p[0:5]
# print(result)

