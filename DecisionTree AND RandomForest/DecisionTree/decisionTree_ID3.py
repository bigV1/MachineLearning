# coding:UTF-8
"""
ID3 decide tree
information divergence
bigV
"""

import pickle
import operator
from math import log
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


# cal  Shannon entropy
def calShannonEnt(dataSet):
    # # -----------计算香农熵的第一种实现方式--------------------
    numEntry = len(dataSet)
    labelsCount = {}
    #the the number of unique elements and their occurance
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelsCount.keys():
            labelsCount[currentLabel] = 0
        labelsCount[currentLabel] +=1
    shannonEnt = 0.0
    for key in labelsCount:
        prob = float(labelsCount[key])/numEntry
        # log2(prob)
        shannonEnt -= prob*log(prob,2)

    # print('---', prob, prob * log(prob, 2), shannonEnt)
    # -----------计算香农熵的第一种实现方式-------------------------

    # # -----------计算香农熵的第二种实现方式------------------------
    # # 统计标签出现的次数
    # label_count = Counter(data[-1] for data in dataSet)
    # # 计算概率
    # probs = [p[1] / len(dataSet) for p in label_count.items()]
    # # 计算香农熵
    # shannonEnt = sum([-p * log(p, 2) for p in probs])
    # # -----------计算香农熵的第二种实现方式-------------------------

    return shannonEnt


# split dataSet
def splitDataSet(dataSet, axis, value):
    """
        dataMat 待划分数据集
        axis 划分数据集的特征
        value 特征的返回值
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    
    # # -----------切分数据集的第二种方式 start-----------
    # retDataSet = [data for data in dataSet for i, v in enumerate(data) if i == axis and v == value]
    # # -----------切分数据集的第二种方式 end-------------

    return retDataSet

# choose best feature to split data
def chooseBestFeatureToSplit(dataSet):
    #the last column is used for the label0s
    numFeature = len(dataSet[0])-1
    baseEntropy = calShannonEnt(dataSet)
    # cal informationentropy
    bestInfoGain = 0.0
    bestFeature = -1
    #iterate over all the features
    for i in range(numFeature):
        #create a list of all the examples of this feature
        featList = [example[i] for example in dataSet]
        #get a set of unique values
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subdataSet = splitDataSet(dataSet, i, value)
            prob = len(subdataSet)/float(len(dataSet))
            newEntropy += prob * calShannonEnt(subdataSet)
        #calculate the info gain; ie reduction in entropy 
        infoGain = baseEntropy - newEntropy
        #compare this to the best gain so far
        if(infoGain > bestInfoGain):
            #if better than current best, set to best
            bestInfoGain = infoGain
            bestFeature = i
    #returns an integer feature index
    return bestFeature

    # # -----------选择最优特征的第二种方式-----------------
    # # 计算初始香农熵
    # base_entropy = calcShannonEnt(dataSet)
    # best_info_gain = 0
    # best_feature = -1
    # # 遍历每一个特征
    # for i in range(len(dataSet[0]) - 1):
    #     # 对当前特征进行统计
    #     feature_count = Counter([data[i] for data in dataSet])
    #     # 计算分割后的香农熵
    #     new_entropy = sum(feature[1] / float(len(dataSet)) * calcShannonEnt(splitDataSet(dataSet, i, feature[0])) \
    #                    for feature in feature_count.items())
    #     # 更新值
    #     info_gain = base_entropy - new_entropy
    #     print('No. {0} feature info gain is {1:.3f}'.format(i, info_gain))
    #     if info_gain > best_info_gain:
    #         best_info_gain = info_gain
    #         best_feature = i
    # return best_feature
    # # -----------选择最优特征的第二种方式-------------

# # 投票表决，获取概率最大的那个分类
# vote
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys:
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

    # # -----------majorityCnt的第二种方式---------------------
    # major_label = Counter(classList).most_common(1)[0]
    # return major_label
    # # -----------majorityCnt的第二种方式-----------------------

# create tree
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # stop splitting when all of the classes are equal
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # #stop splitting when there are no more features in dataSet
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeat]
    # create Tree
    myTree = {bestFeatureLabel:{}} 
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        # recursion
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

# classify 分类
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    print("firstStr:",firstStr)
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat,dict):
        # classify by recursion 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

# save model 存储模型
def storeTree(inputTree,filename):
    fTree = open(filename,'wb')
    pickle.dump(inputTree,fTree)
    fTree.close()
    ### 2
    # with open(filename, 'w') as fTree:
    #     pickle.dump(inputTree, fTree)

# load model 加载模型
def grabTree(filename):
    fTree = open(filename,'rb')
    myTree = pickle.load(fTree)
    return myTree

####====================plot tree=============#####

# 定义文本框 和 箭头格式, sawtooth 波浪方框, round4 矩形方框 , 
# fc表示字体颜色的深浅 
# 决策Node
decisionNode = dict(boxstyle="sawtooth", fc="0.8") 
# 叶子Node
leafNode = dict(boxstyle="round4", fc="0.8")
# arrow_args = dict(arrowstyle="<-")
arrow_args=dict(facecolor='black', arrowstyle="<-")

# 获取叶子数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    # 根节点开始遍历
    for key in secondDict.keys():
        # 判断子节点是否为dict, 不是+1
        if type(secondDict[key]) is dict:
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

# 获取树深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    # 根节点开始遍历
    for key in secondDict.keys():
        # 判断子节点是不是dict, 求分枝的深度
        if type(secondDict[key]) is dict:
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        # 记录最大的分支深度
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

# 画Node
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

# 画出text
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] + cntrPt[0])/2.0 
    yMid = (parentPt[1] + cntrPt[1])/2.0 
    # xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    # yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=10,fontdict={'size': 15, 'color': 'r'})

# 画出树
def plotTree(myTree, parentPt, nodeTxt):
    # 获取叶子节点的数量
    numLeafs = getNumLeafs(myTree)
    # 获取树的深度
    # depth = getTreeDepth(myTree)

    # 找出第1个中心点的位置，然后与 parentPt定点进行划线
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    # print cntrPt
    # 并打印输入对应的文字
    plotMidText(cntrPt, parentPt, nodeTxt)

    firstStr = list(myTree.keys())[0]
    # 可视化Node分支点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 根节点的值
    secondDict = myTree[firstStr]
    # y值 = 最高点-层数的高度[第二个节点位置]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        # 判断该节点是否是Node节点
        if type(secondDict[key]) is dict:
            # 如果是就递归调用[recursion]
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            # 如果不是，就在原来节点一半的地方找到节点的坐标
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            # 可视化该节点位置
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            # 并打印输入对应的文字
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

# 画图
def createPlot(inTree):
    # 创建一个figure的模版
    fig = plt.figure(1, facecolor='white')
    fig.clf()

    axprops = dict(xticks=[], yticks=[])
    # 表示创建一个1行，1列的图，createPlot.ax1 为第 1 个子图，
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)

    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    # 半个节点的长度
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()






####===========================test============================####

# 测试数据集
def retrieveTree(i):
    listOfTrees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return listOfTrees[i]


def createTestData():
    feature = [[1,1,1],
                [1,1,1],
                [1,0,0],
                [0,1,0],
                [0,1,0]]
    labels = ['nop','yep']

    return feature, labels


def fishTest():
    feature, labels = createTestData()
    print("feature:",len(feature[0]))
    shannonEnt = calShannonEnt(feature)
    print(shannonEnt)
    myTree = createTree(feature,labels)
    ## test plotTree
    # myTree = retrieveTree(0)
    createPlot(myTree)

def run():
    # 加载隐形眼镜相关的 文本文件 数据
    fr = open('lenses.txt')
    #fr = open('C:\\Users\\li541\\Desktop\\Meachine Learning\\input\\classify\\lenses.txt')
    # 解析数据，获得 features 数据
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 得到数据的对应的 Labels
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 使用上面的创建决策树的代码，构造预测隐形眼镜的决策树
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    # 画图可视化展现
    # createPlot(lensesTree)

    # classify test
    result = classify(lensesTree,lensesLabels,['young','hyper','yes','normal','hard'])
    print(result)


if __name__=="__main__":
    # fishTest()
    run()    



