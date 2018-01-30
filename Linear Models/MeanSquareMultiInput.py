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

def arrangedata(filename):
    from numpy import genfromtxt
    mydata = genfromtxt(filename, delimiter=',')
    mydata= mydata[:,0:123] 
    return mydata


training1=arrangedata("CandC−train1.csv")
training2=arrangedata("CandC−train2.csv")
training3=arrangedata("CandC−train3.csv")
training4=arrangedata("CandC−train4.csv")
training5=arrangedata("CandC−train5.csv")

test1=arrangedata("CandC−test1.csv")
test2=arrangedata("CandC−test2.csv")
test3=arrangedata("CandC−test3.csv")
test4=arrangedata("CandC−test4.csv")
test5=arrangedata("CandC−test5.csv")


def regression(data):
    result=np.hsplit(data, 123)
    result=np.asarray(result)
    xs = result[0:122,:,0]
    xs=xs.T
    #xs is a 1994 x 123 matrix
    ys = result[122:123,:,0]
    #ys is all the ys in size 1994 x 1
    ys=ys.T
    
    wValue = np.linalg.inv(xs.T.dot(xs)).dot(xs.T).dot(ys)
    return wValue,xs,ys

def mSquareError(w_p,x,y):    
    N=len(y)
    y_p=x.dot(w_p)
   
    Loss= (1/(N))*(sum((y_p-y)**2,0))
       # MSE=2*(Loss/N)**(0.5)    
    return Loss[0]

def regStep(data,test):
    result = regression(data)
    z=np.hsplit(test, 123)
    z=np.asarray(z)
    xs = z[0:122,:,0]
    xs=xs.T
    #xs is a 1994 x 123 matrix
    ys = z[122:123,:,0]
    #ys is all the ys in size 1994 x 1
    ys=ys.T
    MsquareE=mSquareError(result[0],xs,ys)

        
    return MsquareE

mss1=regStep(arrangedata("CandC−train1.csv"),arrangedata("CandC−test1.csv"))
mss2=regStep(arrangedata("CandC−train2.csv"),arrangedata("CandC−test2.csv"))
mss3=regStep(arrangedata("CandC−train3.csv"),arrangedata("CandC−test3.csv"))
mss4=regStep(arrangedata("CandC−train4.csv"),arrangedata("CandC−test4.csv"))
mss5=regStep(arrangedata("CandC−train5.csv"),arrangedata("CandC−test5.csv"))
    
avg=(mss1+mss2+mss3+mss4+mss5)/5


def ridgeRegression(data,z): 
    
    result=np.hsplit(data, 123)
    result=np.asarray(result)
    x = result[0:122,:,0]
    x=x.T
    #xs is a 1994 x 123 matrix
    y = result[122:123,:,0]
    #ys is all the ys in size 1994 x 1
    y=y.T
    w = np.linalg.inv(x.T.dot(x) + z*np.identity(122)).dot(x.T).dot(y)
    return [w,x,y]

def getxy(data):
    result=np.hsplit(data, 123)
    result=np.asarray(result)
    x = result[0:122,:,0]
    x=x.T
    #xs is a 1994 x 123 matrix
    y = result[122:123,:,0]
    #ys is all the ys in size 1994 x 1
    y=y.T
    return [x,y]
    

learntparameters = []
for lm in range(0,200):
    lm=lm*0.01
    w1=ridgeRegression(training1, lm)
    w2=ridgeRegression(training2, lm)
    w3=ridgeRegression(training3, lm)
    w4=ridgeRegression(training4, lm)
    w5=ridgeRegression(training5, lm)
    
    learntparameters.extend([w1[0],w2[0],w3[0],w4[0],w5[0]])
    
    
    m1=mSquareError(w1[0],getxy(test1)[0],getxy(test1)[1])
    m2=mSquareError(w2[0],getxy(test2)[0],getxy(test2)[1])
    m3=mSquareError(w3[0],getxy(test3)[0],getxy(test3)[1])
    m4=mSquareError(w4[0],getxy(test4)[0],getxy(test4)[1])
    m5=mSquareError(w5[0],getxy(test5)[0],getxy(test5)[1])
    bestM=(m1+m2+m3+m4+m5)/5

    print("lm number: "+ str(lm) + " --> " +str(bestM))

bestM=(m1+m2+m3+m4+m5)/5

learntparameters=result=np.asarray(learntparameters)
learntparameters=learntparameters[:,:,0]
learntparameters=learntparameters.T



#best lm number: 0.27 --> 0.0176848782014

    