def BRR(X_train,Y_train,X_test,Y_test,lamda):	
	# -*- coding: utf-8 -*-
	"""
	Created on Mon Mar  5 10:11:45 2018
	
	@author: lily
	"""
	
	from sklearn import linear_model
	import numpy as np
	from sklearn.preprocessing import MinMaxScaler
	#归一化
	Y_train=np.reshape(Y_train,(len(Y_train),1))
	Y_test=np.reshape(Y_test,(len(Y_test),1))
	scaler_x=MinMaxScaler(feature_range=(-1,1))
	X_train=scaler_x.fit_transform(X_train)
	X_test=scaler_x.transform(X_test)
	scaler_y=MinMaxScaler(feature_range=(0,1))
	Y_train=scaler_y.fit_transform(Y_train)
	Y_test=scaler_y.transform(Y_test)
	Y_train=Y_train.ravel()
	Y_test=Y_test.ravel()
	#RVFL
	#需要交叉验证的参数
	lamda=lamda
	#固定参数
	m=500
	p=X_train.shape[1]
	#随机抽取A，b
	A=np.random.uniform(low=-lamda,high=lamda,size=(m,p))
	b=np.random.uniform(low=-lamda,high=lamda,size=(m,1))
	#计算sigmoid矩阵
	h=np.dot(X_train,A.T)+b.T#(n,p)*(p,m)+(1,m)=(n,m)+(1,m)=(n,m)
	h=1.0/(1+np.exp(-h))#(n,m)
	#Bayesian Ridge Regression
	reg=linear_model.BayesianRidge(alpha_1=1e-06,alpha_2=1e-06,lambda_1=1e-06,lambda_2=1e-06,n_iter=300,tol=0.001)
	reg.fit(h,Y_train)
	#对测试集进行相同的操作
	h_test=np.dot(X_test,A.T)+b.T
	h_test=1.0/(1+np.exp(-h_test))
	y_pred=reg.predict(h_test)
	#计算mse,r2,Evar
	y_pred=np.reshape(y_pred,(len(y_pred),1))
	Y_test=np.reshape(Y_test,(len(Y_test),1))
	mse_test=(((y_pred-Y_test)**2).sum())/np.size(Y_test,0)
	Y_mean=np.ones_like(Y_test)*np.mean(Y_test)
	r2=1-(((y_pred-Y_test)**2).sum())/(((Y_test-Y_mean)**2).sum())
	n=len(Y_test)
	err=Y_test-y_pred
	err_mean=np.ones_like(err)*np.mean(err)
	err_var=(((err-err_mean)**2).sum())/(n-1)
	y_var=(((Y_test-Y_mean)**2).sum())/(n-1)
	Evar=1-err_var/y_var
	#输出结果
	return [mse_test,r2,Evar]
