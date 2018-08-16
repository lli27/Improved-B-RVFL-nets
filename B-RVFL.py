#!encoding=utf-8
"""
Created on Thu Nov 23 19:28:01 2017

@author: lily
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import BRR
fr=open(r'delta_elv.txt')
data=[inst.strip().split(',') for inst in fr.readlines()]
for i in range(len(data)):
    data[i]=[float(x) for x in data[i]]
data=np.mat(data)
X=data[:,:-1]
Y=data[:,-1]
Y=np.reshape(Y,(len(Y),1))
lamda=10**(np.linspace(0,1,3))
temp=1000
for i in range(len(lamda)):
	result=[]
	for j in range(10):
		kf = KFold(n_splits=10,shuffle=True)
		for train,test in kf.split(X):
			result.append(BRR.BRR(X[train],Y[train],X[test],Y[test],lamda[i]))
	result=pd.DataFrame(result) 
	re=np.mean(result)[0]#the mean of 'mse_test' in the 10-kfold
	print(lamda[i],re)
	if re<temp:
		temp=re#record the minimum 'mse_test'
		mm=np.mean(result)
		ss=std=np.std(result)
		lamda_=lamda[i]#record the best 'lamda' value
print('\nbest lamda value=\n',lamda_)
print('\nmean=\n',mm)
print('\nstd=\n',ss)
