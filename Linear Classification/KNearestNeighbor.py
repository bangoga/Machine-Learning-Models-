# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 05:41:39 2018

@author: asus
"""

import pandas as pd
import numpy as np  # np is the shortcut to be used. 
import matplotlib.pyplot as plt
import pandas as pd
import os
"for reading the files" 
import csv
import collections
from copy import copy, deepcopy
import math

def arrangedata(filen):
    from numpy import genfromtxt
    data = genfromtxt(filen, delimiter=',')
    

    return data

ds1m0=arrangedata("DS1_m_0.txt");
ds1m1=arrangedata("DS1_m_1.txt");
ds1cov=arrangedata("DS1_Cov.txt");
ds1cov=ds1cov[:,:20]
ds1m0=ds1m0[:20]
ds1m1=ds1m1[:20]


m=np.random.multivariate_normal(ds1m0,ds1cov,2000);
n=np.random.multivariate_normal(ds1m1,ds1cov,2000);

negative=np.array([[-1]] * 2000)
positive=np.array([[1]] * 2000)

#classify
m0 = np.hstack((m, positive))
m1 = np.hstack((n, negative))
   # m1=np.random.multivariate_normal(ds1m1,ds1cov,2000);

np.random.shuffle(m0)
trainingm0, testm0 =(m0)[600:,:], (m0)[:600,:]

np.random.shuffle(m1[0])
trainingm1, testm1 = (m1)[600:,:],(m1)[:600,:]


"ds1 has all randomized test values"
ds1=np.concatenate([testm1,testm0])
np.random.shuffle(ds1)

tr= np.concatenate([trainingm1,trainingm0])
np.random.shuffle(tr)

"Trainm0 has all m0 class 1"
"Trainm0 has all m1 class 2" 


# maximization in pi 

n1=(np.shape(trainingm1))[0]
n2=(np.shape(trainingm0))[0]
piValue=n1/(n1+n2)

test=ds1[:,:20]

" -------------------------[ tr is training set ] ------------------- "

" -------------------------[ test is test set ] ------------------- "



" -------------------------[ ] ------------------- "


#now to actually keep track of the indexes 
"Finds me the euclid value in terms of all features"
def euclidDistance (x,y):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

def nearestK (y,train,k):
    "single Y euclidean"
    dst=[]
    "for all xs"
    for i in range(train.shape[0]):
        temp=euclidDistance(y,train[i,:])
        dst.append((temp,i))
        
    dst = np.asarray(dst)  
        
    #idx = np.argpartition(dst, -k)
    arr = dst[dst[:,0].argsort()]
    return arr[:k,:] #last k elements + index  

def classify(table):
    table=np.array(table)
    newtable=[]
    for i in range(table.shape[0]):
        if (table[i] > 0):
            newtable.append(1)
        if(table[i]<0):
            newtable.append(-1)
    return newtable

def knear(dst,k):
    #arr = dst[dst[:,0].argsort()]
    return dst[len(dst)-k:len(dst)] #last k elements + index 




def accuracy(final,classes):
    acc=0
    for values in range(1200):
        if(final[values]==classes[values]):
           acc=acc+1
    return acc

def precision(final,classes):
    #true positives / true positive+ false positives
    #true postivive is a value if  final value = +1
    t_pos=0
    tf_pos=0
    for values in range(1200):
        if((final[values]==1) and (final[values] == classes[values])):
            t_pos=t_pos+1
        if (final[values]==1):
            tf_pos=tf_pos+1
    
    return t_pos/tf_pos


def recall(final,classes):
    
    
    t_pos=0 # true positive
    tf_negative=0 #true positive + false negative
    for values in range(1200):
        if((final[values]==1) and (final[values] == classes[values])):
            t_pos=t_pos+1
        if ((final[values]==-1) and (final[values] != classes[values])):
            tf_negative=tf_negative+1
    
    tf_negative=tf_negative+t_pos
    return t_pos/tf_negative
















allc=[]
classifiedT=tr[:,20]
TestClass = ds1[:,20]

kn=13
totalInd=[]
#allNeighbours=nearestK (t[1,:],tr,1800)
#allIndices= allNeighbours[:,1]

for j in range(test.shape[0]):
    allNeighbours=nearestK (test[j,:],tr,kn)
    allIndices= allNeighbours[:,1]
    totalInd = np.hstack((totalInd, allIndices))
    
indicMatrix=totalInd.reshape(test.shape[0],kn)
indicMatrix=indicMatrix.T
#add their classified values all and divide by k


classifcationTable=[]
print(indicMatrix.shape[1])
for m in range(indicMatrix.shape[1]):
    total=0
    for i in range(indicMatrix.shape[0]):
        x=indicMatrix[i,m]
        total=total+classifiedT[int(x)]
        total=(1/kn)*total
    classifcationTable.append(total)
    

            
binarytable=classify(classifcationTable)




rec=recall(binarytable,TestClass)
accuracy=accuracy(binarytable,TestClass)/1200
prec = precision(binarytable,TestClass)
fvalue=(2*prec*rec)/(prec+rec)


Truevalues=0
for values in range(1200):
    if binarytable[values] == TestClass[values]:
        Truevalues=Truevalues+1;
        
accuracy=Truevalues/1200
allc.append(accuracy)
