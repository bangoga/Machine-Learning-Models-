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


def sigmoidfunction(x,w1,w0):
    a=x.dot(w1.T)+w0
    return a


def w1cal(co,m1,m2):
    m=m1-m2
    w=np.linalg.inv(co).dot(m)
    return w

"co = covariance"
"m1=mean1"
"m2=mean2"
"lmd = P(c1)"

def w0cal(co,m1,m2,lmd):
    p1=(-0.5)*(m1.T).dot(np.linalg.inv(co).dot(m1))
    p2=(0.5)*(m2.T).dot(np.linalg.inv(co).dot(m2))
    p3=math.log(lmd/(1-lmd))
    
    w0=p1+p2+p3
    
    return w0

#sig functions
def sig(a):
    result=[]
    for i in range(a.shape[0]):
        temp= 1 / (1 + math.exp(-a[i]))
        result.append(temp)
        
    return 1 / (1 + np.exp(-a))



def classify(x):
    result =[]
    x=np.asarray(x)
    for i in range(x.shape[0]):
        if(x[i]<0.5):
            result.append(-1)
        else:
            result.append(1)
            
    return result


"recalculate mean given all of the "
def calm1(x,N1):
    sm= x.sum(axis=0)
    sm=(1/N1)*(sm)
    return sm

def calm2(x,N2):
    sm= x.sum(axis=0)
    sm=(2/N2)*(sm)
    return sm

#maiximize in terms of Covariance

def calCov(m1,m2,x1,x2,n1,n2,n):
    dst=np.zeros((20,20))
    for i in range(x1.shape[0]):
        r=x1[i,:]-m1
        r=r.reshape(20,1)
        value=r.dot(r.T)
        dst=dst+value
        
    s1=(1/n1)*dst
    
    dst2=np.zeros((20,20))
    for i in range(x2.shape[0]):
        r2=x2[i,:]-m2
        r2=r2.reshape(20,1)
        value2=r2.dot(r2.T)
        dst2=dst2+value2
        
    s2=(1/n2)*dst
    
    final = (n1/n)*s1 + (n2/n)*s2
        
    return final
#Maximize in terms of Means 
recalM0=calm1(trainingm0[:,:20],n1)
recalM1=calm1(trainingm1[:,:20],n2)
recalCo = calCov(recalM0,recalM1,trainingm0[:,:20],trainingm1[:,:20],n1,n2,(n1+n2))



# calls
w=(w1cal(recalCo,recalM0,recalM1))
w0=(w0cal(recalCo,recalM0,recalM1,piValue))
    
test=ds1[:,:20]
a = sigmoidfunction(test,w,w0)
classes=ds1[:,20]
 

yes=sig(a)
final=classify(yes)


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


    
rec=recall(final,classes)
accuracy=accuracy(final,classes)/1200
prec = precision(final,classes)
fvalue=(2*prec*rec)/(prec+rec)

        
            




