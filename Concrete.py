# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 19:28:01 2017

@author: lily
"""

import numpy as np
import pandas as pd
import Improved_B_RVFL
from sklearn.model_selection import train_test_split
import xlrd
import random
#data=xlrd.open_workbook('CASP.xlsx')
#data=data.sheets()[0]
#X=np.zeros((45730,9))
#for i in range(9):
#    X[:,i]=data.col_values(i)
#X=X[:20000,:]
#Y=data.col_values(9)
#Y=Y[:20000]
fr=open(r'delta_elv.txt')
data=[inst.strip().split(',') for inst in fr.readlines()]
for i in range(len(data)):
    data[i]=[float(x) for x in data[i]]
data=np.mat(data)
X=data[:,:-1]
Y=data[:,-1]
m=3;k=40000;result=[]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=1)
for i in range(1):
    result.append(Improved_B_RVFL.variational_inference(X_train,Y_train,X_test,Y_test,m,k))
result=pd.DataFrame(result)
print(np.mean(result))
print(np.std(result))
"""
0    0.014456
1    0.017371
2    0.594792
3    0.601574
4    5.000000
"""