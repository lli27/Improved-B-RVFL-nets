# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 19:28:01 2017

@author: lily
"""

import numpy as np#0.0003
import pandas as pd
import Improved_B_RVFL
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold


#fr=open('Concrete_Slump.txt')
#data=[inst.strip().split(',') for inst in fr.readlines()]
#for i in range(len(data)):
#    data[i]=[float(x) for x in data[i]]
#data=np.mat(data)
#X=data[:,1:8]
#Y=data[:,8]
#m=2;k=60000;result=[]
#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=1)
#for i in range(10):
#    result.append(Improved_B_RVFL.variational_inference(X_train,Y_train,X_test,Y_test,m,k))
#result=pd.DataFrame(result)
#print(np.mean(result))
#print(np.std(result))

fr=open('ele1.txt')
data=[inst.strip().split(',') for inst in fr.readlines()]
for i in range(len(data)):
    data[i]=[float(x) for x in data[i]]
data=np.mat(data)
X=data[:,:-1]
Y=data[:,-1]
#m=5;k=60000;result=[]
#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=1)
#for i in range(10):
#    result.append(Improved_B_RVFL.variational_inference(X_train,Y_train,X_test,Y_test,m,k))
#result=pd.DataFrame(result)
#print(np.mean(result))
#print(np.std(result))


#fr=open('baseball.txt')
#data=[inst.strip().split(',') for inst in fr.readlines()]
#for i in range(len(data)):
#    data[i]=[float(x) for x in data[i]]
#data=np.mat(data)
#X=data[:,:-1]
#Y=data[:,-1]
m=3;k=40000;result=[]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=1)
for i in range(1):
    result.append(Improved_B_RVFL.variational_inference(X_train,Y_train,X_test,Y_test,m,k))
result=pd.DataFrame(result)
print(np.mean(result))
print(np.std(result))

#kf = KFold(n_splits=4)
#for train,test in kf.split(X):
    #result.append(Improved_B_RVFL.variational_inference(X[train],Y[train],X[test],Y[test],m,k))
"""
0    0.000076
1    0.000187
2    0.997143
3    0.997221
4    5.000000
"""
