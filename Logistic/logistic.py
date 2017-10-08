import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()#默认删除转移符制表符，然后分割
        #讲三个数据加入到dataMat中，X0=1
        #X1和X2去testSet.txt的每行前两个数据
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))#第三行表示的是标签
    return dataMat,labelMat

def sigmoid(intX):
    return 1.0/(1+np.exp(-intX))

#alpha是移动步长
#maxCycle是训练次数
#

def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)#将数组转变成矩阵，然后就是矩阵相乘
    labelMat = np.mat(classLabels).transpose()#W将标签转化成矩阵之后求转置
    m,n = np.shape(dataMatrix)#返回矩阵的大小
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        t = dataMatrix * weights
        h = sigmoid(t)#h是一个1*m维矩阵
        error = labelMat-h#计算出他们的差
        weights = weights + alpha *dataMatrix.transpose()*error
    return weights

#画出决策边界
def plotBestFit(weights):
    dataMat,labelMat = loadDataSet()#加载数据和标签
    dataArr = np.array(dataMat)#将矩阵转化成数组
    n = np.shape(dataArr)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=np.arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

#随机梯度上升算法变量误差都是数值
def stocGradAscent0(dataMatrix,classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha*error *dataMatrix[i]
    return weights

#改进随机梯度上升算法
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = np.shape(dataMatrix)
    weights=np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            print('randindex',randIndex)
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet),trainingLabels,500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0#读取测试数据的行数
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print('the error rate of this test is %f' % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is: %f' % (numTests,errorSum/(float(numTests))))

if __name__ == '__main__':
    # dataArr,labelMat = loadDataSet()
    # # weights = gradAscent(dataArr,labelMat)
    # # plotBestFit(weights.getA())#将矩阵变成数组
    # weights = stocGradAscent0(np.array(dataArr),labelMat)
    # # print(weights)
    # plotBestFit(weights)
    #改进后的随机梯度算法
    # weights = stocGradAscent1(np.array(dataArr),labelMat)
    # plotBestFit(weights)
    multiTest()