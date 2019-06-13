# -*- coding: utf-8 -*-
#无减枝操作

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 最小二乘损失
def err(dataSet):
    # return sum((dataSet[:,-1]- dataSet[:,-1].mean())**2) # 最原始的写法
    return np.var(dataSet[:, -1]) * dataSet.shape[0]  # 均方差*数据总条数


# 划分数据集，按出入的数据列fea，数据值value将数据划分为两部分
def splitDataSet(dataSet, fea, value):
    dataSet1 = dataSet[dataSet[:, fea] <= value]
    dataSet2 = dataSet[dataSet[:, fea] > value]
    return dataSet1, dataSet2


# 选择最好的特征划分数据集，min_sample每次划分后每部分最少的数据条数，epsilon误差下降阈值，值越小划分的决策树越深
def chooseBestFeature(dataSet, min_sample=4, epsilon=0.5):
    features = dataSet.shape[1] - 1  # x特征列数量
    sErr = err(dataSet)  # 整个数据集的损失
    minErr = np.inf
    bestColumn = 0  # 划分最优列
    bestValue = 0  # 划分最优的值
    nowErr = 0  # 当前平方误差
    if len(np.unique(dataSet[:, -1].T.tolist())) == 1:  # 数据全是一类的情况下 返回
        return None, np.mean(dataSet[:, -1])
    for fea in range(0, features):  # 按x特征列循环
        for row in range(0, dataSet.shape[0]):  # 遍历每行数据，寻找最优划分
            dataSet1, dataSet2 = splitDataSet(dataSet, fea, dataSet[row, fea])  # 获得切分后的数据
            if len(dataSet1) < min_sample or len(dataSet2) < min_sample:  # 按行遍历时总会有一些划分得到的集合不满足最小数据条数约束，跳过此类划分
                continue
            nowErr = err(dataSet1) + err(dataSet2)  # 计算当前划分的平方误差
            # print('fea:',fea,'row:',row,'datavalue',dataSet[row,fea],'nowErr',nowErr)
            if nowErr < minErr:  # 判断获得最优切分值
                minErr = nowErr
                bestColumn = fea
                bestValue = dataSet[row, fea]
        # print('fea',fea,'minErr',minErr,'bestColumn',bestColumn,'bestValue',bestValue)
    if (sErr - minErr) < epsilon:  # 当前误差下降较小时，返回
        return None, np.mean(dataSet[:, -1])
    # 当前最优划分集合
    dataSet1, dataSet2 = splitDataSet(dataSet, bestColumn, bestValue)
    if len(dataSet1) < min_sample or len(dataSet2) < min_sample:  # 如果划分的数据集很小，返回
        return None, np.mean(dataSet[:, -1])
    return bestColumn, bestValue


def createTree(dataSet):
    """
    输入：训练数据集D，特征集A，阈值ε
    输出：决策树T
    """
    bestColumn, bestValue = chooseBestFeature(dataSet)
    if bestColumn == None:  # 所有列均遍历完毕，返回
        return bestValue
    retTree = {}  # 决策树
    retTree['spCol'] = bestColumn  # 最优分割列
    retTree['spVal'] = bestValue  # 最优分割值
    lSet, rSet = splitDataSet(dataSet, bestColumn, bestValue)  # 按当前最优分割列级值划分为左右2枝
    retTree['left'] = createTree(lSet)  # 迭代继续划分左枝
    retTree['right'] = createTree(rSet)  # 迭代继续划分右枝
    return retTree


if __name__ == '__main__':
    # 使用sin函数随机产生x，y数据
    X_data_raw = np.linspace(-3, 3, 50)
    np.random.shuffle(X_data_raw)  # 随机打乱数据
    y_data = np.sin(X_data_raw)  # 产生数据y
    x = np.transpose([X_data_raw])  # 将x进行转换
    y = y_data + 0.1 * np.random.randn(y_data.shape[0])  # 产生数据y，增加随机噪声
    dataSet = np.column_stack((x, y.reshape((-1, 1))))  # 将x与y进行合并

    #    data=pd.read_table('D:/python_data/ex0.txt',header=None)
    #    dataSet=data.values

    mytree = createTree(dataSet)
    print('mytree\n', mytree)
