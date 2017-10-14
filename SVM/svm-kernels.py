import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import random
from os import listdir

def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr=open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return  aj

class optSturct:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.X=dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2)))
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)

def calcEk(oS,k):
    fXk =float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k]+oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    if len(validEcacheList) >1:
        for k in validEcacheList:
            if k== i:
                continue
            Ek = calcEk(oS,k)
            deltaE = abs(Ei-Ek)
            if(deltaE > maxDeltaE):
                maxk = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK,Ej
    else:
        j = selectJrand(i,oS.m)
        Ej = calcEk(oS,j)
    return j,Ej

def updateEk(oS,k):
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0:
            print ("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print ("alpha_j变化太小")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optSturct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler,kTup)
    iter = 0
    entrieSet = True
    alphaPairsChanged = 0
    while ((iter < maxIter and alphaPairsChanged > 0) or entrieSet):
        alphaPairsChanged = 0
        if entrieSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print('全样本遍历:迭代%d次 样本%d,alpha被优化%d次' % (iter, i, alphaPairsChanged))
            iter += 1
        if entrieSet:
            entrieSet = False
        elif alphaPairsChanged == 0:
            entrieSet = True
        print('迭代次数:%d ' % iter)
    return oS.b, oS.alphas

def calcWs(alphas,dataArr,classLabels):
    X=np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m,n=np.shape(X)
    w=np.zeros((n,1))
    for i in range(m):
        w+=np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

#非线性可分，利用核函数
def kernelTrans(X,A,kTup):
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0] == 'lin':
        K=X*A.T
    elif kTup[0] =='rbf':
        for j in range(m):
            deltaRow = X[j,:]-A
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('kernel处碰到了一个问题，内核没有被识别')
    return K

def showClassifer(alphas, dataMat, classLabels, w, b):
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if classLabels[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)  # 正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)  # 负样本散点图
    # 绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    # y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    # plt.plot([x1, x2], [y1, y2])
    #找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()

#利用核函数进行分类函数的向径测试函数
def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    datMat=np.mat(dataArr)
    labelMat=np.mat(labelArr).transpose()
    svInd =np.nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]#构建支持向量矩阵
    labelSV=labelMat[svInd]
    print('支持向量机个数%d '%np.shape(sVs)[0])
    m,n=np.shape(datMat)
    errorCount=0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf',k1))
        predict=kernelEval.T*np.multiply(labelSV,alphas[svInd])+b
        if np.sign(predict)!= np.sign(labelArr[i]):
            errorCount+=1
    print('训练集错误率: %.2f'%(float(errorCount)/m))
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount=0
    dataMat =np.mat(dataArr)
    labelMat =np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf',k1))
        predict = kernelEval.T*np.multiply(labelSV,alphas[svInd])+b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount+=1
    print('测试集错误率 %.2f' % (float(errorCount)/m))
    w = calcWs(alphas, dataArr, labelArr)
    showClassifer(alphas,dataArr,labelArr,w,b)

'''
利用SVM进行手写数字识别
'''

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImage(dirName):
    hwLabels = []
    trainingFileList = listdir(dirName)
    m=len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:] =img2vector('%s/%s'%(dirName,fileNameStr))
    return trainingMat,hwLabels

def testDigits(kTup=('rbf',10)):
    dataArr,labelArr = loadImage('trainingDigits')
    b,alphas = smoP(dataArr,labelArr,200,0.0001,10,kTup)
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A>0)[0]
    sVs = datMat[svInd]
    labelISV = labelMat[svInd]
    print('支持向量机的个数为:%d' %np.shape(sVs)[0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict = kernelEval.T*np.multiply(labelISV,alphas[svInd])+b
        if np.sign(predict)!=np.sign(labelArr[i]):
            errorCount+=1
    print('训练集错误率为:%f' % (float(errorCount)/m))
    dataArr,labelArr = loadImage('testDigits')
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict = kernelEval.T*np.multiply(labelISV,alphas[svInd])+b
        if np.sign(predict) !=np.sign(labelArr[i]):
            errorCount+=1
    print('测试集错误率为:%.2f'%(float(errorCount)/m))



if __name__=='__main__':
    #testRbf()
    testDigits()