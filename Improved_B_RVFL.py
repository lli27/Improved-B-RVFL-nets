# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:42:57 2017

@author: lily
"""
def sigmoid_kernel(X,beta,A,b):
    import theano.tensor as T
    out=T.dot(X,A)+b
    X=1.0/(1+T.exp(-out))
    out=T.dot(X,beta)
    return out
def variational_inference(X_train,Y_train,X_test,Y_test,m,k):
    import numpy as np
    import pymc3 as pm
    from sklearn.preprocessing import MinMaxScaler
    import theano
    import matplotlib.pyplot as plt
    import numpy
    import random
    #输入数据并进行标准化
    n,p=np.shape(X_train)
    Y_train=np.reshape(Y_train,(len(Y_train),1))
    Y_test=np.reshape(Y_test,(len(Y_test),1))
    scaler_x=MinMaxScaler(feature_range=(-1,1))
    X_train=scaler_x.fit_transform(X_train)
    X_test=scaler_x.transform(X_test)
    scaler_y=MinMaxScaler(feature_range=(0,1))
    Y_train=scaler_y.fit_transform(Y_train)
    Y_test=scaler_y.transform(Y_test)
    X_train=theano.shared(X_train)
    sigma=0.1
    rd_num=int(sigma*len(Y_train))
    rd=random.sample(range(len(Y_train)),rd_num)
    sm=np.random.uniform(-0.1,0,size=rd_num)
    Y_train=np.ravel(Y_train)
    Y_train[rd]=sm
#    a1=[];a2=[];a3=[];a4=[];a5=[];bb=[];a6=[];a7=[];a8=[];a9=[];a10=[];a11=[]
#    def tracker(approx,loss_history,i):
#        if (i % 1000) == 0 or i==1:
#            tr=pm.sample_approx(approx=approx,draws=5000)
#            kk=np.mean(tr['A'],axis=0)
#            a1.append(kk[0][0])
#            a2.append(kk[1][0])
#            a3.append(kk[0][1])
#            a4.append(kk[1][1])
#            a5.append(kk[0][2])
#            a6.append(kk[1][2])
#            a7.append(kk[0][3])
#            a8.append(kk[1][3])
#            a9.append(kk[0][4])
#            a10.append(kk[1][4])
#            #a11.append(kk[0][4])
#            bb.append(np.mean(tr['b']))
#        if i==60000:
#            x=np.linspace(0,60000,num=61)
#            plt.plot(x,a1,label='A11')
#            plt.plot(x,a3,label='A21')
#            plt.plot(x,a5,label='A31')
#            plt.plot(x,a7,label='A41')
#            plt.plot(x,a9,label='A51')
#            plt.xlabel('Iteration')
#            plt.ylabel('The mean value of A')
#            plt.grid(True,linestyle='-.',linewidth="1")
#            plt.legend()
#            plt.show()
#            plt.plot(x,a2,label='A12')
#            plt.plot(x,a4,label='A22')
#            plt.plot(x,a6,label='A32')
#            plt.plot(x,a8,label='A42')
#            plt.plot(x,a10,label='A52')
#            plt.xlabel('Iteration')
#            plt.ylabel('The mean value of A')
#            plt.grid(True,linestyle='-.',linewidth="1")
#            plt.legend()
#            plt.show()
#            plt.plot(x,a1,label='A11')
#            plt.plot(x,a2,label='A12')
#            plt.xlabel('Iteration')
#            plt.ylabel('The mean value of A')
#            plt.grid(True,linestyle='-.',linewidth="1")
#            plt.legend()
#            plt.show()
#            plt.plot(x,a3,label='A21')
#            plt.plot(x,a4,label='A22')
#            plt.xlabel('Iteration')
#            plt.ylabel('The mean value of A')
#            plt.grid(True,linestyle='-.',linewidth="1")
#            plt.legend()
#            plt.show()
#            plt.plot(x,bb)
#            plt.ylim(0,2)
#            plt.xlabel('Iteration')
#            plt.ylabel('The mean value of b')
#            plt.grid(True,linestyle='-.',linewidth="1")
#            plt.show()
#        return i
    #定义模型
    basic_model=pm.Model()
    with basic_model:
        b=pm.Normal('b',mu=0,tau=1)
        A=pm.Normal('A',mu=0,tau=1,shape=(p,m))
        #b=b.random(size=m)
        #A=np.reshape(A,(m,p))
        gamma_0=pm.Gamma('gamma_0',alpha=10**(-5),beta=10**(-5))
        gamma_1=pm.Gamma('gamma_1',alpha=10**(-5),beta=10**(-5))
        beta=pm.Normal('beta',mu=0,tau=gamma_0,shape=(m,1))
        Y_obs=pm.Normal('Y_obs',mu=sigmoid_kernel(X_train,beta,A,b),tau=gamma_1,observed=Y_train)
        start=pm.find_MAP()
        #approx=pm.fit(k,start=start,obj_optimizer=pm.adam(),callbacks=[tracker])
        approx=pm.fit(k,start=start,obj_optimizer=pm.adam())
#        plt.xlabel('Iteration')
#        plt.ylabel('Negative ELBO track')
#        plt.grid(True,linestyle='-.',linewidth="1")
#        plt.plot(approx.hist[10000:])
        #在拟合好的模型中，对参数z={beta,A,b,gamma_0,gamma_1}进行抽样
        trace=pm.sample_approx(approx=approx,draws=5000)
        #pm.traceplot(trace)
        #print(trace[-1])
        #print('A=',A)
        #pm.summary(trace)
        #取5000次后验预测的均值为最终结果。
        post_pred=pm.sample_ppc(trace,samples=5000,model=basic_model)
        y_train_pred=np.mean(post_pred['Y_obs'],axis=0)
        #对预测结果与实际结果进行比较。
        mse_train=(((y_train_pred-Y_train)**2).sum())/np.size(Y_train,0)
        X_train.set_value(X_test)
        post_pred=pm.sample_ppc(trace,samples=5000,model=basic_model)
        y_test_pred=np.mean(post_pred['Y_obs'],axis=0)
        mse_test=(((y_test_pred-Y_test)**2).sum())/np.size(Y_test,0)#
    Y_mean=np.ones_like(Y_test)*np.mean(Y_test)
    r2=1-(((y_test_pred-Y_test)**2).sum())/(((Y_test-Y_mean)**2).sum())#
    n=len(Y_test)
    err=Y_test-y_test_pred
    err_mean=np.ones_like(err)*np.mean(err)
    err_var=(((err-err_mean)**2).sum())/(n-1)
    y_var=(((Y_test-Y_mean)**2).sum())/(n-1)
    Evar=1-err_var/y_var #0.9818
    #print('mse_train=',mse_train,'\n mse_test=',mse_test,'\n r2=',r2,'\n Evar=',Evar,'\n m=',m)    
    return mse_train,mse_test,r2,Evar,m
