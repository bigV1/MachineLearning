# -*- coding:UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 加载数据集
dataMat = []
fr = open("data/testSet.txt")
for line in fr.readlines():
    curLine = line.strip().split('\t')
    fltLine = list(map(float,curLine))    # 映射所有的元素为 float（浮点数）类型
    dataMat.append(fltLine)

# 训练模型
# 默认的是kmeans++
km = KMeans(n_clusters=4) # 初始化
km.fit(dataMat) # 拟合
km_pred = km.predict(dataMat) # 预测
centers = km.cluster_centers_ # 质心

# 可视化结果
plt.scatter(np.array(dataMat)[:, 1], np.array(dataMat)[:, 0], c=km_pred)
plt.scatter(centers[:, 1], centers[:, 0], c="r")
plt.show()
