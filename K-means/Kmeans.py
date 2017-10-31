import numpy as np
from numpy import *

'''
将数据从文件中取出来
并且将字符类型转化成浮点型
'''
def loadDataSet(filename):
    dataMat = []
    fr=open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine=list(map(float,curLine))#这里有一个转化问题
        dataMat.append(fltLine)
    return dataMat

'''
计算欧氏距离
函数参数:
    VectA,VectB 两个向量
返回值:
    距离结果
'''
def distEclud(VecA,VecB):
    return np.sqrt(sum(np.power(VecA-VecB,2)))

'''
函数作用引用:
    为给定数据集构建一个包含k个随机质心的集合
函数参数:
    dataSe  - 原始数据集
    k - 质心个数
返回值:
    返回质心的坐标
'''
def randCent(dataSet,k):
    n=np.shape(dataSet)[1]
    centroids=np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])#求出每一列最小值
        rangeJ=float(max(dataSet[:,j])-minJ)#归一化
        #random.rand(k,1)生成一个k*1的随机矩阵，里面的值都在0，1之间
        centroids[:,j]=minJ+rangeJ*np.random.rand(k,1)#随机产生质心
    return centroids#返回质心

#K均值聚类算法
'''
计算出函数的质心点，将数据集进行分类
函数参数:
    dataSet - 
'''
def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    m=shape(dataSet)[0]
    '''
    clusterAssment - 保存质心
    第一维保存引索值、
    第二维保存最小距离的平方和
    '''
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet,k)#随机产生质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])#计算第i个数据与到各个质心的距离
                if distJI < minDist:
                    minDist = distJI#保存最小的距离
                    minIndex = j#保存对应的簇
            if clusterAssment[i, 0] != minIndex:#判断质心是否改变
                clusterChanged = True
                clusterAssment[i, :] = minIndex,minDist**2#更新最小引索，保存最小平方和，即该点属于那一簇
        print(centroids)#打印质心
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A==cent)[0]]#获取该簇中所有的点
            centroids[cent,:] = mean(ptsInClust, axis=0)#将质心更新
    return centroids,clusterAssment#返回质心，最小值以及引索

'''
2分k均值计算
将数据集利用二分法分成多个簇
函数参数:
    dataSet - 数据集
    k - 簇的个数
    distMeas - 距离的方式
返回值:
    质心，引索以及最小欧式距离
'''
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))#和kMeans一样
    #按列求均值，然后变成一行
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]#将数据转换成list
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2#计算每一个到质心的欧氏距离
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0],:]#获取簇为i的所有数据点
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)#将簇i进行二分kmeans划分
            sseSplit = sum(splitClustAss[:, 1])#计算所有的距离的和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])#计算不属于簇i的方差值
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:#更新
                bestCentToSplit = i#最佳切分点
                bestNewCents = centroidMat#簇的矩阵
                bestClustAss = splitClustAss.copy()#将样本进行深复制
                lowestSSE = sseSplit + sseNotSplit#更新最小的值
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList) #求最大的簇
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit#
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],:] = bestClustAss
    return mat(centList), clusterAssment

def showDataSet(dataSet,cluster):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    m=shape(dataSet)[0]
    n=shape(cluster)[0]
    x1 = [];x2=[]
    y1 = [];y2=[]
    for i in range(m):
        x1.append(dataSet[i,0])
        y1.append(dataSet[i,1])
    for j in range(n):
        x2.append(cluster[j,0])
        y2.append(cluster[j,1])
    ax.scatter(x1,y1,marker='o',s=50,c='blue')
    ax.scatter(x2,y2,marker='o',s=90,c='red')
    plt.show()

def distSLC(vecA, vecB):  # Spherical Law of Cosines
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0  # pi is imported with numpy


import matplotlib
import matplotlib.pyplot as plt


def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', \
                      'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()



if __name__=='__main__':
    dataMat=np.mat(loadDataSet('testSet2.txt'))
    # print(min(dataMat[:,0]),'\n',min(dataMat[:,1]))
    # print(max(dataMat[:,0]),"\n",max(dataMat[:,1]))
    # print(randCent(dataMat,2))
    # print(distEclud(dataMat[0],dataMat[1]))
    myCentriods,clusterAssing = kMeans(dataMat,3);
    myCentriods=mat(myCentriods)
    showDataSet(dataMat,myCentriods)
    # print(myCentriods)
    # datMat3 = mat(loadDataSet('testSet2.txt'))
    # centList,myNewAssments = biKmeans(datMat3,3)
    # print(centList)
    #geoResult = geoGrab('1 VA Center','Augusta, ME')
    #clusterClubs()

