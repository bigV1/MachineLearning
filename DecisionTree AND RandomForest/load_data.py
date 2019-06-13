

# load data

import pandas as pd


def load_txt(filename):
    recordlist = []
    fp = open(filename,"rb") 	# 读取文件内容
    content = fp.read()
    fp.close()
    rowlist = content.splitlines() 	# 按行转换为一维表
    recordlist=[row.split("\t") for row in rowlist if row.strip()]	
    dataSet = recordlist
    # labels = labels
    return dataSet

def loadDataSet(fileName):
    # assume last column is target value
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # map all elements to float()
        fltLine = []
        for i in curLine:
           fltLine.append(float(i))
        dataMat.append(fltLine)
    return dataMat



def load_csv(filename):
    pass

def load_excel(filename):
    pass

