from math import log
import operator
import matplotlib.pyplot as plt


def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

def calcShannonEnt(dataSet):#计算香农熵
    numEntris = len(dataSet)#求出数据集的长度
    labelCounts = {}# dict变量，保存键值对
    for featVec in dataSet:
        #print(featVec)
        currentLabel = featVec[-1]#表示读取featVec最后一项,等于featVec[len(featVec)-1]
        if currentLabel not in labelCounts.keys():#判断currentlabel是否在labelCount的键里面
            labelCounts[currentLabel] = 0#如果不在，统计为0
        labelCounts[currentLabel]+=1#在的情况下就+1
    shannoEnt=0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntris#prob表示该公式出现的概率
        shannoEnt -= prob*log(prob,2)
    return shannoEnt

def splitDataSet(dataSet,axis,value):#
    reDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:#如果第axia个元素等于value，则将value剔除重新组成数据集
            reduceFeatVec = featVec[:axis]#将featVec从0开始取出axis个值，取括号
            reduceFeatVec.extend(featVec[axis+1:])#从axis+1开始，一直取到最后一个值
            reDataSet.append(reduceFeatVec)#append和extend的区别
    return reDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy=calcShannonEnt(dataSet)#计算香浓熵
    bestInfoGain = 0.0
    bestFeature =-1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]#将每一个数据集的第i位取出来
        uniqueVals = set(featList)#获取其中无重复的元素
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            #划分之后为一个元素集合
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy+=prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy#增益信息，熵 - 条件熵
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):#统计关键字出现的次数
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote]+=1
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        print(sortedClassCount)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    label_copy=labels[:]
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):#
        return classList[0]
    if len(dataSet[0]) == 1:#没有特征停止划分
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)#求出现次数最多的类别
    bestFeatLabel = label_copy[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (label_copy[bestFeat])#删除掉出现次数最多的类别的标签
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = label_copy[:]#获取其他的标签，继续进行分类,进行深拷贝
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

decisitonNode = dict(boxstyle="sawtooth",fc="0.8")
leafNode = dict(boxstyle= "round4",fc="0.8")
arraw_args = dict(arrowstyle="<-")

def classify(inputTree,featLabels,testVec):

    firstStr = list(inputTree.keys())[0]
    #print(firstStr)
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ =='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree,filename):
    import pickle    #pickle序列化对象
    fw=open(filename,'wb')
    pickle.dump(inputTree,fw,0)
    fw.close()

def grabTree(filename):
    import pickle#取出之前序列化的对象
    fr = open(filename,'rb')
    return pickle.load(fr)

if __name__ == '__main__':
    myDat,labels=createDataSet()
    #print(myDat)
    #print(calcShannonEnt(myDat))
    #print(splitDataSet(myDat,0,0))
    #print(chooseBestFeatureToSplit(myDat))
    myTree = createTree(myDat,labels)
    # storeTree(myTree,'classifierstorage.txt')
    # print(grabTree('classifierstorage.txt'))
    #预测隐形眼镜的类型
    classify(myTree,labels,[1,1])
