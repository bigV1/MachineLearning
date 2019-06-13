# -*- coding: utf-8 -*-
"""
SOM网络，自组织特征映射神经网络
Kohonen映射
bigV
"""

from numpy import *
import matplotlib.pyplot as plt


class Kohonen(object):
    def __init__(self):
        self.lratemax = 0.8  # 最大学习率
        self.lratemin = 0.05  # 最小学习率
        self.rmax = 5.0  # 最大聚类半径
        self.rmin = 0.5  # 最小聚类半径
        self.Steps = 1000  # 迭代次数
        self.lratelist = []
        self.rlist = []
        self.w = []
        self.M = 2  # 二维聚类网格参数
        self.N = 2  # 二维聚类网格参数
        self.dataMat = []
        self.classLabel = []

    def loadDataSet(self, fileName):  # 加载数据文件
        numFeat = len(open(fileName).readline().split('\t')) - 1
        fr = open(fileName)
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            lineArr.append(float(curLine[0]));
            lineArr.append(float(curLine[1]));
            self.dataMat.append(lineArr)
        self.dataMat = mat(self.dataMat)

    # 数据标准化(归一化):		# 标准化
    def normalize(self, dataMat):
        [m, n] = shape(dataMat)
        for i in range(n - 1):
            dataMat[:, i] = (dataMat[:, i] - mean(dataMat[:, i])) / (std(dataMat[:, i]) + 1.0e-10)
        return dataMat

    # 计算矩阵各向量之间的距离--欧氏距离
    def distEclud(self, matA, matB):
        ma, na = shape(matA)
        mb, nb = shape(matB)
        rtnmat = zeros((ma, nb))
        for i in range(ma):
            for j in range(nb):
                rtnmat[i, j] = linalg.norm(matA[i, :] - matB[:, j].T)
        return rtnmat

    # 学习率和学习半径函数
    def ratecalc(self, indx):
        lrate = self.lratemax - (float(indx) + 1.0) / float(self.Steps) * (self.lratemax - self.lratemin)
        r = self.rmax - (float(indx) + 1.0) / float(self.Steps) * (self.rmax - self.rmin)
        return lrate, r

    # 初始化第二层网格
    def init_grid(self):
        k = 0;  # 构建第二层网格模型
        grid = mat(zeros((self.M * self.N, 2)))
        for i in range(self.M):
            for j in range(self.N):
                grid[k, :] = [i, j]
                k += 1
        return grid

    # 主算法
    def train(self):
        # 1 构建输入层网络
        dm, dn = shape(self.dataMat)
        normDataset = self.normalize(self.dataMat)  # 归一化数据x
        # 2 构建分类网格
        grid = self.init_grid()  # 初始化第二层分类网格
        # 3 构建两层之间的权重向量
        self.w = random.rand(dn, self.M * self.N);  # 随机初始化权值 w
        distM = self.distEclud  # 确定距离公式
        # 4 迭代求解
        if self.Steps < 10 * dm:    self.Steps = 10 * dm  # 设定最小迭代次数
        for i in range(self.Steps):
            lrate, r = self.ratecalc(i)  # 计算学习率和分类半径
            self.lratelist.append(lrate);
            self.rlist.append(r)
            # 随机生成样本索引，并抽取一个样本
            k = random.randint(0, dm)
            mySample = normDataset[k, :]

            # 计算最优节点：返回最小距离的索引值
            minIndx = (distM(mySample, self.w)).argmin()
            d1 = ceil(minIndx / self.M)  # 计算最近距离在第二层矩阵中的位置
            d2 = mod(minIndx, self.M)
            distMat = distM(mat([d1, d2]), grid.T)
            nodelindx = (distMat < r).nonzero()[1]  # 根据学习距离获取邻域内左右节点
            # 更新权重列
            for j in range(shape(self.w)[1]):
                if sum(nodelindx == j):
                    self.w[:, j] = self.w[:, j] + lrate * (mySample[0] - self.w[:, j])
        # 分配类别标签
        self.classLabel = list(range(dm))
        for i in range(dm):
        	self.classLabel[i] = distM(normDataset[i, :], self.w).argmin()
        self.classLabel = mat(self.classLabel)

    def showCluster(self, plt):
        lst = unique(self.classLabel.tolist()[0])  # 去重
        # 绘图
        i = 0
        for cindx in lst:
            myclass = nonzero(self.classLabel == cindx)[1]
            xx = self.dataMat[myclass].copy()
            if i == 0:
                plt.plot(xx[:, 0], xx[:, 1], 'bo')
            elif i == 1:
                plt.plot(xx[:, 0], xx[:, 1], 'rd')
            elif i == 2:
                plt.plot(xx[:, 0], xx[:, 1], 'gD')
            elif i == 3:
                plt.plot(xx[:, 0], xx[:, 1], 'c^')
            i += 1
        plt.show()

    # 绘制趋势线: 可调整颜色
    def TrendLine(self, plt, mylist, color='r'):
        X = linspace(0, len(mylist), len(mylist))
        Y = mylist
        plt.plot(X, Y, color)
        plt.show()


# 加载数据文件
def loadDataSet(fileName):  # general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1  # get number of fields
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        lineArr.append(float(curLine[0]));
        lineArr.append(float(curLine[1]));
        dataMat.append(lineArr)
    return dataMat


# 显示绘制图形
def displayplot():
    plt.show()


# 绘制二维数据集坐标散点图:无分类
# 适用于 List 和 Matrix
def drawScatter(dataMat, flag=True):
    if type(dataMat) is list:
        px = (mat(dataMat)[:, 0]).tolist()
        py = (mat(dataMat)[:, 1]).tolist()
    if type(dataMat) is matrix:
        px = (dataMat[:, 0]).tolist()
        py = (dataMat[:, 1]).tolist()
    plt.scatter(px, py, c='blue', marker='o')
    if flag: displayplot()


# 路径
def drawPath(Seq, dataMat, color='b', flag=True):
    px = (dataMat[Seq, 0]).tolist()[0]
    py = (dataMat[Seq, 1]).tolist()[0]
    px.append(px[0]);
    py.append(py[0])
    plt.plot(px, py, color)
    if flag: displayplot();


# 绘制二维数据集坐标散点图:有分类
# 适用于 List 和 Matrix
def drawClassScatter(dataMat, classLabels, flag=True):
    # 绘制list
    if type(dataMat) is list:
        i = 0
        for mydata in dataMat:
            if classLabels[i] == 0:
                plt.scatter(mydata[1], mydata[2], c='blue', marker='o')
            else:
                plt.scatter(mydata[1], mydata[2], c='red', marker='s')
            i += 1
    # 绘制Matrix
    if type(dataMat) is matrix:
        i = 0
        for mydata in dataMat:
            if classLabels[i] == 0:
                plt.scatter(mydata[0, 1], mydata[0, 2], c='blue', marker='o')
            else:
                plt.scatter(mydata[0, 1], mydata[0, 2], c='red', marker='s')
            i += 1
    if flag: displayplot();


# 绘制分类线
def ClassifyLine(begin, end, weights, flag=True):
    # 确定初始值和终止值,精度
    X = linspace(begin, end, (end - begin) * 100)
    # 建立线性分类方差
    Y = -(float(weights[0]) + float(weights[1]) * X) / float(weights[2])
    plt.plot(X, Y, 'b')
    if flag: displayplot()


# 绘制趋势线: 可调整颜色
def TrendLine(X, Y, color='r', flag=True):
    plt.plot(X, Y, color)
    if flag: displayplot()


# 合并两个多维的Matrix，并返回合并后的Matrix
# 输入参数有先后顺序
def mergMatrix(matrix1, matrix2):
    [m1, n1] = shape(matrix1)
    [m2, n2] = shape(matrix2)
    if m1 != m2:
        print("different rows,can not merge matrix")
        return
    mergMat = zeros((m1, n1 + n2))
    mergMat[:, 0:n1] = matrix1[:, 0:n1]
    mergMat[:, n1:(n1 + n2)] = matrix2[:, 0:n2]
    return mergMat


# 计算相关系数
# 入口: x,y 的元素个数必须相同的一维数组
# start 开始计算相关的下标 >=0
# 返回: 相关系数
def corref(x, y, start=0):
    N = len(x)
    if (N != len(y)) or (N < start + 2):
        return 0.0
    Sxx = Syy = Sxy = Sx = Sy = 0
    for i in range(start, N):
        Sx = Sx + x[i]
        Sy = Sy + y[i]
    Sx = Sx / (N - start)
    Sy = Sy / (N - start)
    for i in range(start, N):
        Sxx = Sxx + (x[i] - Sx) * (x[i] - Sx)
        Syy = Syy + (y[i] - Sy) * (y[i] - Sy)
        Sxy = Sxy + (x[i] - Sx) * (y[i] - Sy)
    r = abs(Sxy) / math.sqrt(Sxx * Syy)
    return r




def runTest():
    # 加载坐标数据文件
    SOMNN = Kohonen()
    dataSet = loadDataSet("dataset.txt")
    dataMat = mat(dataSet)
    dm, dn = shape(dataMat)
    # 归一化数据
    normDataset = SOMNN.normalize(dataMat)

    # 参数
    # 学习率
    rate1max = 0.8  # 0.8
    rate1min = 0.05
    # 学习半径
    r1max = 3
    r1min = 0.8  # 0.8

    ## 网络构建
    Inum = 2
    M = 2
    N = 2
    K = M * N  # Kohonen总节点数

    # Kohonen层节点排序
    k = 0
    jdpx = mat(zeros((K, 2)))
    for i in range(M):
        for j in range(N):
            jdpx[k, :] = [i, j]
            k = k + 1

    # 权值初始化
    w1 = random.rand(Inum, K)  # 第一层权值

    ## 迭代求解
    ITER = 200
    for i in range(ITER):

        # 自适应学习率和相应半径
        rate1 = rate1max - (i + 1) / float(ITER) * (rate1max - rate1min)
        r = r1max - (i + 1) / float(ITER) * (r1max - r1min)
        # 随机抽取一个样本
        k = random.randint(0, dm)  # 生成样本的索引,不包括最高值
        myndSet = normDataset[k, :]  # xx

        # 计算最优节点：返回最小距离的索引值
        minIndx = (SOMNN.distEclud(myndSet, w1)).argmin()
        d1 = ceil(minIndx / M)
        d2 = mod(minIndx, N)
        distMat = SOMNN.distEclud(mat([d1, d2]), jdpx.transpose())
        nodelindx = (distMat < r).nonzero()[1]
        for j in range(K):
            if sum(nodelindx == j):
                w1[:, j] = w1[:, j] + rate1 * (myndSet.tolist()[0] - w1[:, j])

    # 学习阶段
    classLabel = list(range(dm))
    for i in range(dm):
        classLabel[i] = SOMNN.distEclud(normDataset[i, :], w1).argmin()
    # 去重
    lst = unique(classLabel)
    print(lst)
    classLabel = mat(classLabel)
    # 绘图
    i = 0
    for cindx in lst:
        myclass = nonzero(classLabel == cindx)[1]
        xx = dataMat[myclass].copy()
        if i == 0:
            plt.plot(xx[:, 0], xx[:, 1], 'bo')
        if i == 1:
            plt.plot(xx[:, 0], xx[:, 1], 'r*')
        if i == 2:
            plt.plot(xx[:, 0], xx[:, 1], 'gD')
        if i == 3:
            plt.plot(xx[:, 0], xx[:, 1], 'c^')
        i += 1
    plt.show()


def runTest02():
    SOMNN = Kohonen()
    # 加载坐标数据文件
    dataSet = loadDataSet("dataset.txt")
    dataMat = mat(dataSet)
    # print dataMat
    normDataset = SOMNN.normalize(dataMat)
    # print normDataset

    # 生成int随机数，不包含高值
    # print random.randint(0,30)

    # 计算向量中最小值的索引值
    xx = mat([1, 9])
    w1 = mat([[1, 2, 3, 4], [5, 6, 7, 8]])
    minIndx = SOMNN.distEclud(xx, w1).argmin()

    # 计算距离
    jdpx = mat([[0, 0], [0, 1], [1, 0], [1, 1]])
    d1 = ceil(minIndx / 4)
    d2 = mod(minIndx, 4)
    mydist = SOMNN.distEclud(mat([d1, d2]), jdpx.transpose())
    print(mydist)


if __name__ == '__main__':
    #runTest()
    runTest02()