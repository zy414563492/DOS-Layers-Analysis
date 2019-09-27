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
import pickle


dataFile = './Filled_STA/average_STA.mat'
data = scio.loadmat(dataFile)

for i, key in enumerate(data.keys()):
    print(i, key)

average = data['dif_gray_image']
print(type(average))
print(average.shape)
print(average)

with open('matData.pkl', 'wb') as f:
    pickle.dump(average, f)
