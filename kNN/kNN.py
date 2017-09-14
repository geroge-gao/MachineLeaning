from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

def classify0(inX, dataSet, labels, k):
    dataSetSize=dataSet.shape[0]#返回dataset的第一维的长度
    print(dataSetSize)
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    #计算出各点离原点的距离
    #表示diffMat的平方
    sqDiffMat = diffMat**2#平方只针对数组有效
    sqDistances=sqDiffMat.sum(axis = 1)
    distances=sqDistances**0.5
    sortedDistIndices = distances.argsort()#返回从小到大的引索
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]#找到对应的从小到大的标签
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])#numpy里面的数组，注意和list的区别
    labels=['A','A','B','B']
    return group,labels

def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    print(numberOfLines)
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index = 0
    for lines in arrayOLines:
        lines = lines.strip()
        listFromLine = lines.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def show(datingDataMat,datingLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],15.0*array(datingLabels),15.0*array(datingLabels))
    plt.show()


def autoNorm(dataSet):#将特征值归一化
    minVals=dataSet.min(0)#选择数据集中最小的
    maxVals=dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet=zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet-tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio = 0.50  # hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
            print( "the total error rate is: %f" % (errorCount / float(numTestVecs)))
           # print(errorCount)

def img2vector(filename):
    returnVect = zeros((1, 1024))
    print("returnVect\n"+returnVect)
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    print(trainingMat)
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))



if __name__ == "__main__":
    group,labels = createDataSet()
    classer=classify0([0,0],group,labels,3)
    handwritingClassTest()
  #   datingDataMat, datingLabels=file2matrix('datingTestSet2.txt')
  #   show(datingDataMat,datingLabels)






