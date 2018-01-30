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

#"This would read the file and arrange them as arrays from the csv files 
def arrangedata(filename):
    from numpy import genfromtxt
    mydata = genfromtxt(filename, delimiter=',')
    mydata= mydata[:,0:2]
    return mydata

# To plot and visualize the fit:
    
def plotmygraph(data,w_p,title_g):
    x = data[:,0:1]
    y = data[:,1:2]
    xf=np.arange(-1,1,0.01)
    
#    y_p=np.asarray([(x[p,0]**p)*w_p[p,0] for p in range(50)])

    result =  [i ** p for p in range(0,21)  for i in xf]
    result=np.asarray(result)
    result = np.split(result,21)
    result=np.asarray(result)
    x_p=result.T    
    y_p=x_p.dot(w_p)
    
    plt.title(title_g)
    plt.xlabel('x value')
    plt.ylabel('y value')
    plt.plot(x,y,'ro')
    plt.plot(xf,y_p, 'g-')
    plt.show()
    return y_p

#   polynomial fitting using (x^T x)^-1 * x^T Y

def q1(data):
    x = data[:,0:1]
    y = data[:,1:2]

    result =  [i ** p for p in range(0,21)  for i in x]
    result=np.asarray(result)
    result = np.split(result,21)
    result=np.asarray(result)
    result = result[:,:,0]
    x=result.T
    
    value1 = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
   
    return value1


def mSquareError(data,w_p):
    x = data[:,0:1]
    y = data[:,1:2]
    N=len(x)
    
    result =  [i ** p for p in range(0,21)  for i in x]
    result=np.asarray(result)
    result = np.split(result,21)
    result=np.asarray(result)
    x_p=result.T
    
    y_p=x_p.dot(w_p)
    y_p=y_p[0,:,:]
    
    summed=0
    for i in range(N):
        summed=summed+(y_p[i]-y[i])**2
        
    summed=summed*(1/N)
            
    
   # MSE=2*(Loss/N)**(0.5)
    
    return summed

q1data=arrangedata('Dataset_1_train.csv')
q1data2=arrangedata('Dataset_1_valid.csv')
q1data3=arrangedata('Dataset_1_test.csv')

w_p = q1(q1data)
y_p=plotmygraph(q1data,w_p,"Non Regularized training set")

trainingMSE=mSquareError(q1data,w_p)
validMSE=mSquareError(q1data2,w_p)

def Regularize(data,z):
    #identity i = np.eye(4)
    #z being my lambda
    x = data[:,0:1]
    y = data[:,1:2]
    
    result =  [i ** p for p in range(0,21)  for i in x]
    result=np.asarray(result)
    result = np.split(result,21)
    result=np.asarray(result)
    result = result[:,:,0]
    x=result.T
    
    value1 = np.linalg.inv(x.T.dot(x) + z*np.identity(21)).dot(x.T).dot(y)
    
    return value1

def stepregularize(data,data2):
    allLoss=[]
    for x in range(1000):
        z=0.001*x
        w=Regularize(data,z)
        Regloss=[mSquareError(data2,w)]
        allLoss.extend(Regloss)
    
    return allLoss

        
allloss=stepregularize(q1data,q1data2)    
allloss2=stepregularize(q1data,q1data)   

allz=[]
for x in range(1000):
    allz.extend([0.001*x])
    
xf=np.arange(-25,25,0.051)
a=allloss[1:982]
b=allloss2[1:982]


plt.title("Mean Square Error for training and validation")
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.plot(xf,allloss, 'g-',label='valid')
plt.plot(xf,allloss2,'r-',label='training')
plt.legend(loc='upper right')
plt.show()
    


w=Regularize(q1data,0.02)
plotmygraph(q1data,w,"Regularized")
Regloss=mSquareError(q1data3,w) ## loss with test values








