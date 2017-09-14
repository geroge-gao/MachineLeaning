import kNN
import numpy

def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent filter miles earned per year"))
    iceCream=float(input("liters of ice cream consumed per year"))
    datingDataMat,datingLabels=kNN.file2matrix('datingTestSet2.txt')
    normMat ,ranges ,minVals =kNN.autoNorm(datingDataMat)
    inArr = numpy.array([ffMiles,percentTats,iceCream])
    classifierResult =kNN.classify0((inArr-minVals)/ranges,normMat,datingDataMat,3)
    print("you will probably like this person: ".resultList[classifierResult-1])

