# C4.5
# 信息增益率

import pickle
import operator
from numpy import *
import math
import copy
import matplotlib.pyplot as plt
import plotTreeModel

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def loadDataSet(path, labels):
    recordlist = []
    fp = open(path, "r")
    content = fp.read()
    fp.close()
    rowlist = content.splitlines()
    recordlist = [row.split("\t") for row in rowlist if row.strip()]
    dataSet = recordlist
    labels = labels


def train(dataSet,labels):

    labels = copy.deepcopy(labels)
    tree = buildTree(dataSet, labels)


def buildTree(dataSet, labels):
    cateList = [data[-1] for data in dataSet]
    if cateList.count(cateList[0]) == len(cateList):
        return cateList[0]
    if len(dataSet[0]) == 1:
        return maxCate(cateList)
    bestFeat, featValueList = getBestFeat(dataSet)
    bestFeatLabel = labels[bestFeat]
    tree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    for value in featValueList:
        subLabels = labels[:]
        splitDataset = splitDataSet(dataSet, bestFeat, value)
        subTree = buildTree(splitDataset, subLabels)
        tree[bestFeatLabel][value] = subTree
    return tree


def maxCate(catelist):
    items = dict([(catelist.count(i), i) for i in catelist])
    return items[max(items.keys())]


def getBestFeat(dataSet):
    Num_Feats = len(dataSet[0][:-1])
    totality = len(dataSet)
    BaseEntropy = computeEntropy(dataSet)
    ConditionEntropy = []  # 初始化条件熵
    slpitInfo = []  # for C4.5, calculate gain ratio
    allFeatVList = []
    for f in range(Num_Feats):
        featList = [example[f] for example in dataSet]
        [splitI, featureValueList] = computeSplitInfo(featList)
        allFeatVList.append(featureValueList)
        slpitInfo.append(splitI)
        resultGain = 0.0
        for value in featureValueList:
            subSet = splitDataSet(dataSet, f, value)
            appearNum = float(len(subSet))
            subEntropy = computeEntropy(subSet)
            resultGain += (appearNum / totality) * subEntropy
        ConditionEntropy.append(resultGain)  # 总条件熵
    infoGainArray = BaseEntropy * ones(Num_Feats) - array(ConditionEntropy)
    infoGainRatio = infoGainArray / array(slpitInfo)  # c4.5, info gain ratio
    bestFeatureIndex = argsort(-infoGainRatio)[0]
    return bestFeatureIndex, allFeatVList[bestFeatureIndex]


def computeSplitInfo(featureVList):
    numEntries = len(featureVList)
    featureVauleSetList = list(set(featureVList))
    valueCounts = [featureVList.count(featVec) for featVec in featureVauleSetList]
    # caclulate shannonEnt
    pList = [float(item) / numEntries for item in valueCounts]
    lList = [item * math.log(item, 2) for item in pList]
    splitInfo = -sum(lList)
    return splitInfo, featureVauleSetList


def computeEntropy(dataSet):
    datalen = float(len(dataSet))
    cateList = [data[-1] for data in dataSet]
    items = dict([(i, cateList.count(i)) for i in cateList])
    infoEntropy = 0.0
    for key in items:
        prob = float(items[key]) / datalen
        infoEntropy -= prob * math.log(prob, 2)
    return infoEntropy


def splitDataSet(dataSet, axis, value):
    rtnList = []
    for featVec in dataSet:
        if featVec[axis] == value:
            rFeatVec = featVec[:axis]
            rFeatVec.extend(featVec[axis + 1:])
            rtnList.append(rFeatVec)
    return rtnList


def predict(inputTree, featLabels, testVec):
    root = list(inputTree.keys())[0]
    secondDict = inputTree[root]
    featIndex = featLabels.index(root)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]  #
    if isinstance(valueOfFeat, dict):
        classLabel = predict(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)




def createTestData():
    feature = [[1, 1, 1],
               [1, 1, 1],
               [1, 0, 0],
               [0, 1, 0],
               [0, 1, 0]]
    labels = ['nop', 'yep']

    return feature, labels


def fishTest():
    feature, labels = createTestData()
    print("feature:", len(feature[0]))
    shannonEnt = computeEntropy(feature)
    print(shannonEnt)
    myTree = buildTree(feature, labels)
    ## test plotTree
    # myTree = retrieveTree(0)
    plotTreeModel.createPlot(myTree)


def run():
    # 加载隐形眼镜相关的 文本文件 数据
    # fr = open('lenses.txt')
    fr = open('lenses.txt')
    # 解析数据，获得 features 数据
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 得到数据的对应的 Labels
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 使用上面的创建决策树的代码，构造预测隐形眼镜的决策树
    lensesTree = buildTree(lenses, lensesLabels)
    print(lensesTree)
    # 画图可视化展现
    # createPlot(lensesTree)

    # classify test
    result = predict(lensesTree, lensesLabels, ['young', 'hyper', 'yes', 'normal', 'hard'])
    print(result)


if __name__ == "__main__":
    # fishTest()
    run()







