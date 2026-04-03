
import numpy as np
from collections import defaultdict

def loadTrainingData(benchmark,path):
    trainFile=path+"/"+benchmark+".train.rating"
    print(trainFile)

    trainSet=defaultdict(list)
    max_u_id=-1
    max_i_id=-1

    for line in open(trainFile):
        userId,itemId,rating=line.strip().split('\t')
        userId=int(userId)
        itemId=int(itemId)
        rating=float(rating)
        if rating>0:
            if itemId not in trainSet[userId]:
                trainSet[userId].append(itemId)
        max_u_id=max(userId,max_u_id)
        max_i_id=max(itemId,max_i_id)

    for u,i_list in trainSet.items():
        i_list.sort()
    userCount=max_u_id+1
    itemCount=max_i_id+1
    print(userCount)
    print(itemCount)
    print("Training data loading done: %d users, %d items" % (userCount, itemCount))
    #pdb.set_trace()
    return trainSet, userCount, itemCount

def loadTestData(benchmark,path):
    testFile = path + "/" + benchmark + ".test.rating"
    print(testFile)
    testSet = defaultdict(list)
    for line in open(testFile):
        userId, itemId, rating= line.strip().split('\t')
        userId = int(userId)
        itemId = int(itemId)
        rating=float(rating)
        if rating>0:
            testSet[userId].append(itemId)
    return testSet
def to_Vectors(trainSet, userCount, itemCount, userList_test,trainData):
    batchCount=itemCount

    itemCount=userCount
    userCount=batchCount
    batchCount=itemCount

    trainDict=defaultdict(lambda:[0]*userCount)


    for userId,i_list in trainSet.items():
        for itemId in i_list:
            #trainDict[userId][itemId]=trainData[userId][itemId]
            trainDict[userId][itemId] = 1
    trainVector=[]
    for batchId in range(batchCount):
        trainVector.append(trainDict[batchId])
    return np.array(trainVector),batchCount




