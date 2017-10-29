from numpy import *
import matplotlib.pyplot as plt

class treeNode():
    def __init__(self,feat,val,right,left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch  = left


def loadDataSet(filename):
    datMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 注意python2.x和3.x的区别，3.xmap返回的不是list，所以要自
        #强制转换成list，这里是一个大坑啊
        fltLine = list(map(float,curLine))
        datMat.append(fltLine)
    return datMat

'''
binSplitDataSet函数作用:
    将数据集切分，并且将结果返回
函数参数:
    dataSet - 数据集
    feature - 待切分特征
    value - 该特征的某个值
返回值:
    返回切分的两个子集
'''
def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[nonzero(dataSet[:,feature]>value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature]<=value)[0],:]
    return mat0,mat1

#负责生成叶子结点，并且作为变量
def regLeaf(dataSet):
    return mean(dataSet[:,-1])

'''
var是均方差函数
计算总方差=方差乘以样本数
求数据的方差通过决策树的划分将比较靠近的数据划分到一起
'''
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

'''
用最佳的方式切分数据集，并生成叶子节点
Parameters:
    dataSet - 数据集
    leafTYpe - 建立叶子结点的函数
    errType - 误差计算函数
    ops   [容许误差下降值，最小切分数]
returns:
    bestFeat index
    bestFeat value
'''
def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    #
    tolS = ops[0]
    tolN=ops[1]
    # 将数据转换成一维然后进行计算
    if(set(dataSet[:,-1].T.tolist()[0]))==1:#如果全部数据为同一个类别，不用继续划分
        return None,leafType(dataSet)
    m,n=shape(dataSet)
    S=errType(dataSet)#计算数据集方差和
    bestS = inf
    bestIndex = 0
    bestValue = 0
    #循环处理每一列对应的feature
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):#这一部分树上代码不能直接运行
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if(shape(mat0)[0]<tolN or (shape(mat1)[0]<tolN)):#判断切分元素数量是否符合预期
                continue
            newS = errType(mat0)+errType(mat1)#计算切分之后的方差和
            if newS<bestS:#如果值比较小的话，就更新
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if(S - bestS)<tolS:#如果比阀值更小，结束划分
        return None,leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    if(shape(mat0)[0]<tolS) or(shape(mat1)[0]<tolN):#如果划分集合过小，也不划分
        return None,leafType(dataSet)
    return bestIndex,bestValue

'''
函数参数:
    dataSet - 数据集
    leafType - 建立叶子节点的函数
    errType - 计算方差函数
    ops -  [容许样本的下降值 切分的最少样本数]
返回值:
    reTree  - 决策树最后的结果    
'''

def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat ==None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet =binSplitDataSet(dataSet,feat,val)#大于在右边，小于在左边，分为两个数据集
    #递归进行调用，在左右子树中继续递归生成树
    retTree['left'] = createTree(lSet,leafType,errType,ops)#
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree

def showDataSet(dataArr):
    datMat = mat(dataArr)
    x = [];
    y = []
    n=shape(datMat)[0]
    for i in range(n):
        x.append(datMat[i][0])
        y.append(datMat[i][-1])
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=30, c='black', marker='s')
    plt.show()

#判断一个节点类型是否为字典
def isTree(obj):
    return (type(obj).__name__=='dict')

#计算左右节点的均值
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

#检查是否合适并分支
'''
从上而下找到叶子节点，用测试节点来判断这些叶子节点并是否能降低测试误差
函数参数:
    tree - 待剪枝的树
    testData - 剪枝所需要的测试数据
返回值:
    tree - 剪枝完成的树
'''
def prune(tree,testData):
    #判断测试数据集是否为空，如果为空，则返回的tree的均值
    if shape(testData)[0] == 0:
        return getMean(tree)
    #如果分支是字典，就将数据切分
    if(isTree(tree['right']) or isTree(tree['left'])):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    #递归对左子树剪枝
    if isTree(tree['left']):
        tree['left']=prune(tree['left'],lSet)
    #递归对有子树剪枝
    if isTree(tree['right']):
        tree['right']=prune(tree['right'],rSet)

    '''
    如果左右子树都不是字典，也就是说左右都是叶子节点而不是子树，将测试数据集进行分割
    计算总方差和该结果集不被分支的总方差
        如果合并总方差小于不被合并的总方差，那么就合并
    返回值:
        如果可以合并，结果从字典变成了数据            
    '''
    if not isTree(tree['right']) and not isTree(tree['left']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1]-tree['left'],2))+\
            sum(power(rSet[:,-1]-tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1]-treeMean,2))
        #如果合并只有方差小于总方差
        if errorMerge<errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree

#将数据集编程自变量X,和目标变量Y，并且得到回归系数

def linearSolve(dataSet):
    m,n=shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))
    #X的0列为1，为了计算平衡误差
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular,cannot do inverse,\n\
                        try increasing the second value of ops')
    ws = xTx.I*(X.T*Y)#最小二乘法求最优解
    return ws,X,Y

#得到回归系数ws
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

#计算线性模型的误差
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X*ws
    return sum(power(Y-yHat,2))

#回归树按例测试
def regTreeEval(model,inDat):
    return float(model)

'''
# 模型树测试案例
# 对输入数据进行格式化处理，在原数据矩阵上增加第0列，元素的值都是1，
# 也就是增加偏移值，和我们之前的简单线性回归是一个套路，增加一个偏移量

对模型树进行预测
函数参数:
    model - 输入模型 - 回归树模型或者 模型树模型
    inDat - 输入测试数据
返回值:
    得到预测值，并且将其转换成浮点数
'''
def modelTreeEval(model,inDat):
    n = shape(inDat)[1]
    X= mat(ones((1,n+1)))
    X[:,1:n+1] = inDat
    return float(X*model)

# 计算预测的结果
# 在给定树结构的情况下，对于单个数据点，该函数会给出一个预测值。
# modelEval是对叶节点进行预测的函数引用，指定树的类型，以便在叶节点上调用合适的模型。
# 此函数自顶向下遍历整棵树，直到命中叶节点为止，一旦到达叶节点，它就会在输入数据上
# 调用modelEval()函数，该函数的默认值为regTreeEval()

"""
    Desc:
        对特定模型的树进行预测，可以是 回归树 也可以是 模型树
    Args:
        tree -- 已经训练好的树的模型
        inData -- 输入的测试数据
        modelEval -- 预测的树的模型类型，可选值为 regTreeEval（回归树） 或 modelTreeEval（模型树），默认为回归树
    Returns:
        返回预测值
"""

def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree,inData)
    if inData[tree['spInd']]>tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)


def createForeCast(tree,testData,modelEval = regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0]=treeForeCast(tree,mat(testData[i]),modelEval)
    return yHat


if __name__=="__main__":
    # testMat = mat(eye(4))
    # mat0,mat1=binSplitDataSet(testMat,1,0.5)
    # myDat = loadDataSet('ex2.txt')
    # myDat=mat(myDat)
    # myTree=createTree(myDat)
    # print(myTree)
    # myDat2=loadDataSet('ex2.txt')
    # myMat2=mat(myDat2)
    # # myTree = createTree(myMat2,ops=(10000,4))
    # myTree = createTree(myMat2,ops=(0,1))
    # myDat2Test = loadDataSet('ex2test.txt')
    # myMat2Test = mat(myDat2Test)
    # print(prune(myTree,myMat2Test))
    # myMat2 = mat(loadDataSet('exp2.txt'))
    # myTree=createTree(myMat2,modelLeaf,modelErr)
    # print(myTree)
    trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree = createTree(trainMat,ops=(1,20))
    yHat = createForeCast(myTree,testMat[:,0])
    print(corrcoef(yHat,testMat[:,1],rowvar=0)[0,1])
    #再创建一颗模型树
    myTree = createTree(trainMat,modelLeaf,modelErr,(1,20))
    yHat = createForeCast(myTree,testMat[:,0],modelTreeEval)
    print(corrcoef(yHat,testMat[:,1],rowvar=0)[0,1])


