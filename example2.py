# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 19:28:01 2017

@author: lily
"""

import numpy as np
import pandas as pd
import Improved_B_RVFL
from sklearn.model_selection import train_test_split

fr=open('baseball.txt')
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
