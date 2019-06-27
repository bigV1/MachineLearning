from numpy import *
import operator

class KNNClassifier():
    def __init__(self):
        self.dataSet = []
        self.labels = []

    def loadDataSet(self,filename):
        fr = open(filename)
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataLine = list()
            for i in lineArr:
                dataLine.append(float(i))
            label = dataLine.pop() # pop the last column referring to  label
            self.dataSet.append(dataLine)
            self.labels.append(int(label))

    def setDataSet(self, dataSet, labels):
        self.dataSet = dataSet
        self.labels = labels

    def predict(self, data, k):
        self.dataSet = array(self.dataSet)
        self.labels = array(self.labels)
        self._normDataSet()
        dataSetSize = self.dataSet.shape[0]
        # get distance
        diffMat = tile(data, (dataSetSize,1)) - self.dataSet
        sqDiffMat = diffMat**2
        distances = sqDiffMat.sum(axis=1)
        # get K nearest neighbors
        sortedDistIndicies = distances.argsort()
        classCount= {}
        for i in range(k):
            voteIlabel = self.labels[sortedDistIndicies[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        # get fittest label
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def _normDataSet(self):
        minVals = self.dataSet.min(0)
        maxVals = self.dataSet.max(0)
        ranges = maxVals - minVals
        normDataSet = zeros(shape(self.dataSet))
        m = self.dataSet.shape[0]
        normDataSet = self.dataSet - tile(minVals, (m,1))
        normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
        self.dataSet = normDataSet

def test():
    KNN = KNNClassifier()
    dataSet = array([[1.0,1.1],[1.0,1.0],[0.9,0.9],[0,0],[0,0.1],[0,0.2]])
    labels = [1,1,1,2,2,2]
    print(KNN.predict([1.0,1.1], 2))

if __name__ == '__main__':
    KNN = KNNClassifier()
    KNN.loadDataSet('data/testData.txt')
    print(KNN.predict([72011, 4.932976, 0.632026], 5) )
    # KNN.test()
