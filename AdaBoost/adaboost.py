from numpy import *
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    datMat = np.matrix([[1.,2.1],
                       [2.,1.1],
                       [1.3,1.],
                       [1.,1.],
                       [2.,1.]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

def showDataSet(datMat,classLabels):
    xcord0 = [];ycord0 = []
    xcord1 = [];ycord1 = []
    colors = []
    makers = []
    for i in range(len(classLabels)):
        if classLabels[i]==1.0:
            xcord0.append(datMat[i,0])
            ycord0.append(datMat[i,1])
        else:
            xcord1.append(datMat[i,0])
            ycord1.append(datMat[i,1])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord0,ycord0,marker='o',s=90,c='blue')
    ax.scatter(xcord1,ycord1,marker='o',s=90,c='red')
    plt.title('DataSet')
    plt.show()

'''
单层决策树分类函数
函数参数:
    dataMatrix - 数据矩阵
    dimen -  第dimen列，也就是第几个特征
    threshVal - 阈值
    threshIneq - 标志
返回值:
    retArray - 分类结果
'''
def stumpClassify(datMatrix,dimen,threshVal,threshIneq):
    retArray = np.ones((np.shape(datMatrix)[0],1))
    if threshIneq =='lt':#初始化retArray为1
        retArray[datMatrix[:,dimen]<= threshVal] = -1.0#如果小于阈值，则赋值为-1
    else:
        retArray[datMatrix[:,dimen]>threshVal] = -1.0#如果大于阈值，则赋值为-1
    return retArray

'''
找到数据集上最佳的单层决策树
函数参数:
    dataArr - 数据矩阵
    classLabels - 数据标签
    D - 样本权重
返回值:
    bestStump - 最佳单层决策树
    minError - 最小误差
    besetClasEst - 最佳的分类器    
'''

def buildStump(dataArr,classLabels,D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n=np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClaEst = np.mat(np.zeros((m,1)))
    minError = np.inf  #最小误差初始化为正无穷大
    for i in range(n):#遍历所有特征
        rangeMin = dataMatrix[:,i].min()#找出特征的最小值
        rangeMax = dataMatrix[:,i].max()#找出特征的最大值
        stepSize = (rangeMax-rangeMin)/numSteps#计算步长
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:#设置标志
                threshVal = (rangeMin+float(j)*stepSize)#计算阈值
                predictVals = stumpClassify(dataMatrix,i,threshVal,inequal)#计算分类的结果
                errArr = np.mat(np.ones((m,1)))#初始化误差矩阵
                #判断计算值和实际是否有误差，
                #如果没有误差可以则设置为0，若有误差则设置为1
                errArr[predictVals==labelMat]=0#
                weightedError = D.T*errArr#计算误差
                print('split: dim %d,thresh % .2f,thresh inequal: %s,the weighted error is %.3f' %(i,threshVal,inequal,weightedError))
                if weightedError<minError:#找到最小的误差分类
                    minError = weightedError
                    bestClaEst=predictVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClaEst

'''
使用AdaBoost算法提升弱分类器性能
函数参数:
    dataArr - 数据矩阵
    classLabels - 数据标签
    numIt - 最大迭代次数
返回值:
    weakClassArr - 训练好的样本
    aggClassEst - 类别估计累计值
'''

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m=np.shape(dataArr)[0]
    D=np.mat(np.ones((m,1))/m)#初始化权值
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#构建单层决策树
        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))#计算弱分类器的权值alpha应为分母权值不能为0
        bestStump['alpha'] = alpha#单层决策树里面保存权值
        weakClassArr.append(bestStump)#储存单层决策树
        #print("ClassEst: ",classEst.T)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst)#计算e的指数项

        D=np.multiply(D,np.exp(expon))
        D=D/D.sum()#更新样本权值公式
        aggClassEst +=alpha*classEst    #计算类别估计值
        #print("aggClassEst",aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst)!=np.mat(classLabels).T,np.ones((m,1)))#计算误差
        errorRate = aggErrors.sum()/m
        #print('total error: ',errorRate,"\n",)
        if errorRate == 0.0:
            break
    return weakClassArr,aggClassEst

'''
AdaBoost分类器
函数参数:
    datToClass - 待分类样例
    classifierArr - 训练好的分类器
返回值:
    分类结果
'''
def adaClassify(datToClass,classifierArr):
    datMatrix = np.mat(datToClass)
    m=np.shape(datToClass)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):#遍历所有的分类器
        #构建单层决策树
        classEst = stumpClassify(datMatrix,classifierArr[i]['dim'],\
            classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst #将分类器进行组装
    return np.sign(aggClassEst)

#自适应数据加载函数
def load_data(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr=open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#ROC曲线的绘制以及AUC的计算
def plotROC(predStrengths,classLables):
    cur = (1.0,1.0)
    ySum = 0.0
    numPosClas = sum(np.array(classLables)==1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLables)-numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig=plt.figure()
    fig.clf()
    ax=plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLables[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum+=cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ",ySum*xStep)



if __name__=='__main__':
    datMat,classLabels = loadDataSet()
    #showDataSet(datMat,classLabels)
    D=np.mat(np.ones((5,1))/5)
    print(buildStump(datMat,classLabels,D))
    #print(adaBoostTrainDS(datMat,classLabels,9))
    # datMat,labelsArr = load_data('horseColicTraining2.txt')
    # classifierArr,aggClassEst = adaBoostTrainDS(datMat,labelsArr,10)
    # testArr,testLabelsArr = load_data('horseColicTest2.txt')
    # prediction10 = adaClassify(t estArr,testLabelsArr)
    # errArr = np.mat(np.ones((67,1)))
    # print(errArr[prediction10!=np.mat(testLabelsArr).T].sum())
    #plotROC(aggClassEst.T,labelsArr)

