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
    
    
data=arrangedata('Dataset_2_train.csv')


## Calculation Loss function.  L(w0,w1)  = 1/2N * Sum {Y(x;w) - y}^2
## wj = wj - x * l(w0,w1)
def steps(data,val,iterations,stepsize):
    "convergence wj-wj=0"
    w0=0;
    w1=0;
    
    x = data[:,0:1]
    y = data[:,1:2]

    y_p=[w0+w1*xi  for xi in x]
    N=len(x)
    y_p=np.asarray(y_p)
     
    TotalMSE1=[]
    TotalMSE2=[]
    for a in range(iterations):
        y_p=[w0+w1*xi  for xi in x]
        y_p=np.asarray(y_p)
        ## calculate mse as well 
        TotalMSE1.extend(MSE(data,w0,w1))
        TotalMSE2.extend(MSE(val,w0,w1))
        for i in range(N):
            w0=w0-(stepsize*(y_p[i]-y[i]))
            w1=w1-(stepsize*(y_p[i]-y[i])*x[i])
    
    return [w0,w1,TotalMSE1,TotalMSE2];


def plotmygraph(data,w0,w1,ep,lm):
    x = data[:,0:1]
    y = data[:,1:2]
    xf=np.arange(-1,1,0.01)
    
    y_p=[w0+w1*xi  for xi in x]
    y_p=np.asarray(y_p)
    stg="Linear Fit on training Epoch " + str(ep) +", Stepsize: " +str(lm)
    
    plt.title(stg)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x,y,'ro')
    plt.plot(x,y_p, 'g-')
    plt.show()

    return y_p
    
    
    
def MSE(data,w0,w1):
    x = data[:,0:1]
    y = data[:,1:2]
    N=len(x)
    y_p=[w0+w1*xi  for xi in x]
    msError=(1/(N))*(sum((y-y_p)**2,0)) #1/n Sum(y-y*)^2
    return msError


val=arrangedata('Dataset_2_valid.csv')

loss = steps(data,val,10000,1e-6) ## These values changed for question 2
allX=[]    
for i in range(10000):
    allX.extend([i])   

plt.title("Learning Curve Visualization")
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.plot(allX,loss[3],'bs',label='valid')
plt.plot(allX,loss[2], 'g-',label='train')
plt.legend(loc='upper right')
plt.show()

#5 different visualizations
yp=plotmygraph(data,loss[0],loss[1],10000,1e-6) 

## The Best stepspize was done by manuallly replacing the values over a smaller iteration number
## The below will show graph fits of 0.0001


loss = steps(data,val,10,1e-4) ## These values changed for question 2
allX=[]    
for i in range(10):
    allX.extend([i])
    
yp=plotmygraph(data,loss[0],loss[1],10,1e-4) 



loss = steps(data,val,50,1e-4) ## These values changed for question 2
allX=[]    
for i in range(50):
    allX.extend([i])
    
yp=plotmygraph(data,loss[0],loss[1],50,1e-4) 

loss = steps(data,val,100,1e-4) ## These values changed for question 2
allX=[]    
for i in range(100):
    allX.extend([i])
    
yp=plotmygraph(data,loss[0],loss[1],100,1e-4)

loss = steps(data,val,1000,1e-4) ## These values changed for question 2
allX=[]    
for i in range(1000):
    allX.extend([i])
    
yp=plotmygraph(data,loss[0],loss[1],1000,1e-4)

loss = steps(data,val,10000,1e-4) ## These values changed for question 2
allX=[]    
for i in range(10000):
    allX.extend([i])
    
yp=plotmygraph(data,loss[0],loss[1],10000,1e-4)


#report test MSE for best step size
test=arrangedata('Dataset_2_test.csv')
testMSE=MSE(test,loss[0],loss[1])
yp=plotmygraph(test,loss[0],loss[1],10000,1e-4) 

