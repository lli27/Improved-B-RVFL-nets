# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:25:09 2017

@author: lily
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#fr=open('Concrete_Slump.txt')
#data=[inst.strip().split(',') for inst in fr.readlines()]
#for i in range(len(data)):
#    data[i]=[float(x) for x in data[i]]
#data=np.mat(data)
#X=data[:,1:8]
#Y=data[:,8]
#fr=open('baseball.txt')
#data=[inst.strip().split(',') for inst in fr.readlines()]
#for i in range(len(data)):
#    data[i]=[float(x) for x in data[i]]
#data=np.mat(data)
#X=data[:,:-1]
#Y=data[:,-1]
fr=open('delta_elv.txt')
data=[inst.strip().split(',') for inst in fr.readlines()]
for i in range(len(data)):
    data[i]=[float(x) for x in data[i]]
data=np.mat(data)
X=data[:,:-1]
Y=data[:,-1]

result=[]
for i in range(100):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=1)
    Y_train=np.reshape(Y_train,(len(Y_train),1))
    Y_test=np.reshape(Y_test,(len(Y_test),1))
    scaler_x=MinMaxScaler(feature_range=(-1,1))
    X_train=scaler_x.fit_transform(X_train)
    X_test=scaler_x.transform(X_test)
    scaler_y=MinMaxScaler(feature_range=(0,1))
    Y_train=scaler_y.fit_transform(Y_train)
    Y_test=scaler_y.transform(Y_test)
    clf=RandomForestRegressor(n_estimators=100)#40,60,50,100
    #clf=svm.SVR(gamma=10**(-2))#-0.5,-2,-0.5,-2
    Y_train=np.ravel(Y_train)
    Y_test=np.ravel(Y_test)
    clf.fit(X_train,Y_train)
    y_train_pred=clf.predict(X_train)
    mse_train=(((y_train_pred-Y_train)**2).sum())/np.size(y_train_pred,0)
    y_test_pred=clf.predict(X_test)
    mse_test=(((y_test_pred-Y_test)**2).sum())/np.size(y_test_pred,0)
    Y_mean=np.ones_like(Y_test)*np.mean(Y_test)
    r2=1-(((y_test_pred-Y_test)**2).sum())/(((Y_test-Y_mean)**2).sum())#
    n=len(Y_test)
    err=y_test_pred-Y_test
    err_mean=np.ones_like(err)*np.mean(err)
    err_var=(((err-err_mean)**2).sum())/(n-1)
    y_var=(((Y_test-Y_mean)**2).sum())/(n-1)
    Evar=1-err_var/y_var #0.9818
   #print('mse_test=',mse_test,'\n r2=',r2,'\n Evar=',Evar)
    a=[mse_train,mse_test,r2,Evar]
    result.append(a)
result=pd.DataFrame(result)
print(np.mean(result))
print(np.std(result))
