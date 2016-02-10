__author__ = 'Krunal'
import csv
import random
import math
import operator
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix
import random
import datetime

import scipy as sp
from scipy.spatial import distance
from math import*

class kNN(object):
    def testkNN(self,trainX,trainY,testX,k):
        #print(datetime.datetime.now())
        p=[]
        for y in range(len(testX)): # iterate for loop for every test point
            dist=[]
            dist=np.linalg.norm(trainX-testX[y],axis=1) # find eucliean distance
            temp=zip(dist,trainY) # zip with label
            sortedtemp = sorted(temp,key=lambda tup: tup[0])
            Nearestneighbors=sortedtemp[:k] # find k nearest neighbour
            NearestneighborsClass = map(operator.itemgetter(1), Nearestneighbors)
            c=Counter(NearestneighborsClass).most_common() # find most common class label
            p.append(c[0][0]) # append into prediction array
        return p

    def modelaccuracy(self,testY, predictiontestY): # check accuracy of model
        tp = 0
        for x in range(len(testY)):
            if testY[x][-1] == predictiontestY[x]:
                tp += 1 # if its is matching then increasing count
        total = float(len(testY))
        return (tp/total) * 100.0 # actual / total divide

    def condensedata(self,trainX, trainY):
        index= [] # dummy variable
        SS=[] # declare  forSubset train X
        SSL=[] # declare for subset train Y
        temp=[]
        condensedIdx=[] # declare final array to store condensed indices
        index = range(len(trainX))
        SS.append(trainX[0])
        SSL.append(trainY[0])
        while sum(index): # execute till all element of train X
            nonzero_index = np.nonzero(index) # finding remaining train X
            RI=random.choice(nonzero_index[0]) # Select random index from remaining train X
            index[RI]=0 # Make index zero if it is used once
            temp=[]
            temp.append(trainX[RI])
            predictedtestY = self.testkNN(SS,SSL,temp,1) # execute 1NN for given subset
            if(predictedtestY[0] != trainY[RI]): # if its matching
                SS.append(trainX[RI]) # append if not matching bcz its required
                SSL.append(trainY[RI])
                condensedIdx.append(RI) # store that index into final array
        return condensedIdx

#####################################################################HELER CODE ############
nTrain = 5000
nTest = 15000
k=3
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data', header=None)
#df = pd.read_csv('C://Users/Krunal/Documents/DSBA/Spring 2016/ML/letter-recognition.data.txt', header=None)
trainX = np.array(df.iloc[0:nTrain,1:]) # split array
trainY = np.array(df.iloc[0:nTrain,0])
testX = np.array(df.iloc[nTest:,1:]) # split test data
testY = np.array(df.iloc[nTest:,0])
condensedtrainX = []
condensedtrainY = []
time1=datetime.datetime.now()
knn = kNN()
print 'Train set: ' + repr(len(trainX))
print 'Test set: ' + repr(len(testX))
print 'K Value :' + repr(k)
#knn.testkNN1(trainX,trainY,testX,k)
predictedtestY1 = knn.testkNN(trainX,trainY,testX,k)  # EXECUTE ENTIRE kNN
#print(predictedtestY1)
accuracy=knn.modelaccuracy(testY,predictedtestY1) # find accuracy
time2=datetime.datetime.now()
t=time2-time1
print('Time with entire training set:')
print(t) # print time
print('Accuracy with entire training set')
print(accuracy) # print accuracy
lab=np.unique(trainY)
print(lab)
c=confusion_matrix(testY,predictedtestY1,labels=lab)
print(c)
print('############################################# Condensed ')
print 'Train set: ' + repr(len(trainX))
print 'Test set: ' + repr(len(testX))
print 'K Value :' + repr(k)
time3=datetime.datetime.now()
condensedIdx=knn.condensedata(trainX,trainY)
for i in condensedIdx:# find condensed training set from new condesed indices
    condensedtrainX.append(trainX[i])
    condensedtrainY.append(trainY[i])

condensedtrainXtemp = np.array(condensedtrainX)

#print(condensedIdx)
print(len(condensedIdx))
predictedtestY = knn.testkNN(condensedtrainXtemp,condensedtrainY,testX,k) # execute kNN on condensed train Set
accuracy1=knn.modelaccuracy(testY,predictedtestY)
time4=datetime.datetime.now()
t1=time4-time3
print('Time with Condensed training set')
print(t1)
print('Accuracy with Condensed training set')
print(accuracy1)

