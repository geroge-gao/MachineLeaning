from numpy import *
import re
import feedparser
import operator

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createVocablist(dataset):
    vocaset=set([])#创建一个空集
    for document in dataset:
        vocaset = vocaset | set(document)#创建两个集合合并的并集
    return list(vocaset)

def setOfWord2Vec(vocablist,inputset):#第一个参数为词汇表，第二个参数为文档或者文件
    returnVec = [0]*len(vocablist)#创建一个所有向量为0的元素
    for word in inputset:#将文档中对应出现的单词记为1
        if word in vocablist:
            returnVec[vocablist.index(word)] = 1
        else:
            print("the word %s is not in my vocabulary!"%word)
    return returnVec

def trainNB0(trainMaxtrix,trainCategory):
    numTrainDocs = len(trainMaxtrix)
    numWords =len(trainMaxtrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] ==1:
            p1Num+=trainMaxtrix[i]
            p1Denom+=sum(trainMaxtrix[i])
        else:
            p0Num+=trainMaxtrix[i]
            p0Denom+=sum(trainMaxtrix[i])
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass):
    p1 = sum(vec2Classify*p1Vec)+log(pClass)
    p0 = sum(vec2Classify*p0Vec)+log(pClass)
    if p1>p0:
        return 1
    else:
        return 0

def testNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocablist(listOPosts)
    train_mat = []
    for postinDoc in listOPosts:
        train_mat.append(setOfWord2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(train_mat),array(listClasses))
    testEntry = ['love','my','dalmation']
    this_doc = array(setOfWord2Vec(myVocabList,testEntry))
    print(testEntry,"classified as: ",classifyNB(this_doc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    this_doc = array(setOfWord2Vec(myVocabList,testEntry))
    print(testEntry,'classified as: ',classifyNB(this_doc,p0V,p1V,pAb))

#朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

#使用朴素贝叶斯进行交叉认证
def textParse(bigString):
    listOfTokens = re.split(r'\W*',bigString)
    return {tok.lower() for tok in listOfTokens if len(tok)>2}

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocablist(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorcount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorcount += 1
            print("classification error",docList[docIndex])
    print('the error rate is: ',float(errorcount)/len(testSet))



#RSS源分类及高频词去出函数
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(),key = operator.itemgetter(1),reverse=True)#python3中舍弃了iteritems
    return sortedFreq[:30]

def localWords(feed1,feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocablist(docList)
    top30Words = calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    error_count = 0
    for docIndex in testSet:
        word_vector = bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(word_vector),p0V,p1V,pSpam) != classList[docIndex]:
            error_count += 1
    print('the error rate is: ',float(error_count)/len(testSet))
    return vocabList,p0V,p1V

#显示地域相关用词
def getTopWords(ny,sf):
    vocabList,p0V,p1V = localWords(ny,sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF,key=lambda pair:pair[1],reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY,key= lambda pair:pair[1],reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY")
    for item in sortedNY:
        print(item[0])


if __name__=="__main__":
    # listOPosts,listClasses = loadDataSet()
    # myVocabList = createVocablist(listOPosts)
    # returnVec = setOfWord2Vec(myVocabList,listOPosts[0])
    # train_mat = []
    # for psot_in_doc in listOPosts:
    #     train_mat.append(setOfWord2Vec(myVocabList,psot_in_doc))
    #     p0V,p1V,pAb = trainNB0(train_mat,listClasses)
    # testNB()
    # mysent ='This book is that best book on python or M.L. I have ever laid eye upon'
    # print(mysent.split())
    # regEx = re.compile('\\W*')
    # listOfTokens = regEx.split(mysent)
    # print(listOfTokens)
    #spamTest()
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    vocabList,pSF,pNY = localWords(ny,sf)
    vocabList,pSF,pNY = localWords(ny,sf)
    vocabList,pSF,pNY = localWords(ny,sf)
    getTopWords(ny,sf)


