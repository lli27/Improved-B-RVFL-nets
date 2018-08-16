#!encoding=utf-8
import numpy as np
from scipy.linalg.misc import norm
from sklearn import linear_model
import time
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# 读取数据
#fr=open(r'D:\python3_work\baseball.txt')
#data=[inst.strip().split(',') for inst in fr.readlines()]
#for i in range(len(data)):
#    data[i]=[float(x) for x in data[i]]
#data=np.mat(data)
#X=data[:,:-1]
#Y=data[:,-1]
#fr=open(r'D:/python3_work/Concrete_Slump.txt')
#data=[inst.strip().split(',') for inst in fr.readlines()]
#for i in range(len(data)):
#    data[i]=[float(x) for x in data[i]]
#data=np.mat(data)
#X=data[:,1:8]
#Y=data[:,8]
fr=open(r'ele1.txt')
data=[inst.strip().split(',') for inst in fr.readlines()]
for i in range(len(data)):
    data[i]=[float(x) for x in data[i]]
data=np.mat(data)
X=data[:,:-1]
Y=data[:,-1]
#import xlrd
#data=xlrd.open_workbook('Concrete.xls')
#data=data.sheets()[0]
#X=np.zeros((1030,8))
#for i in range(8):
#    X[:,i]=data.col_values(i)
#Y=data.col_values(8)
#fr=open(r'delta_elv.txt')
#data=[inst.strip().split(',') for inst in fr.readlines()]
#for i in range(len(data)):
#    data[i]=[float(x) for x in data[i]]
#data=np.mat(data)
#X=data[:,:-1]
#Y=data[:,-1]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=1)
scaler_x=MinMaxScaler(feature_range=(-1,1))
X_train=scaler_x.fit_transform(X_train)
X_test=scaler_x.transform(X_test)
scaler_y=MinMaxScaler(feature_range=(0,1))
Y_train=scaler_y.fit_transform(Y_train)
Y_test=scaler_y.transform(Y_test)
Y_train=Y_train.ravel()
sigma=0.1
rd_num=int(sigma*len(Y_train))
rd=random.sample(range(len(Y_train)),rd_num)
sm=np.random.uniform(-1,0,size=rd_num)
Y_train[rd]=sm
x_tr=X_train;y_tr=Y_train;x_te=X_test;y_te=Y_test.ravel()
mse_te=1000000;num=0;result=[]
while num<1:
    # 可变参数
    #print('num=',num)
    num=num+1
    L_max = 100  # 隐层神经元个数
    eps = 0.001
    T_max = 10  # 随机配置次数
    lambdaScalar = np.arange(1, 5, 0.5)
    #lambdaScalar = np.array([1,5, 15, 30, 50, 100, 150, 200])
    # lambda_ = 2**7
    r_init = 0.1  # 0<r<1
    Tem = 2  # 循环次数
    tauTimeMax = 5

    # 默认参数
    e_0 = copy.deepcopy(y_tr)
    e_0_te = copy.deepcopy(y_te)
    d = x_tr.shape[1]
    L = 0
    beta = []
    yHat_tr = 0
    yHat_te = 0


    def con2(e):  # new temp
        sum_ = norm(e)
        output = sum_
        return output


    def funcXi(w_vector, b, input, e_1, L, r):  # new
        mu = (1 - r) / float(L + 1)
        sum = 0
        featureMap = np.dot(input, w_vector) + b
        g = 1. / (1 + np.exp(-featureMap))
        g = g.reshape(len(g))
        output = (np.dot(e_1, g) ** 2) / float((np.dot(g, g))) - (1 - r - mu) * np.dot(e_1, e_1)
        return output


    def funcBeta(g, e_1T):
        # print x
        g = g.reshape(len(g))
        beta = np.dot(e_1T, g) / float((np.dot(g, g)))
        return beta


    while L <= L_max and con2(e_0) > eps and Tem < 5000:
        e_1 = copy.deepcopy(e_0)
        e_1_te = copy.deepcopy(e_0_te)

        for lambda_ in lambdaScalar:
            empty = 0
            tauTime = 1
            k = 1
            xi_temp = 0
            r=r_init
            #lambdaOK = 0
            #print('现在lambda_=',lambda_,'隐层神经元个数L=',L)
            #####步骤4到16####头
            while tauTime <= tauTimeMax and k <= T_max:
                # print 'k',k
                w = np.random.uniform(-lambda_, lambda_, size=d)
                b = np.random.uniform(-lambda_, lambda_, size=1)
                w_vector = w.reshape((len(w), 1))

                xi = funcXi(w_vector, b, x_tr, e_1, L, r)
                if xi >= 0:
                    empty = 1
                    if xi >= xi_temp:
                        xi_temp = xi
                        w_vector_temp = w_vector
                        b_temp = b
                k = k + 1
                if k == T_max + 1 and empty == 0:
                    #print('******************************')
                    # time.sleep(1)
                    #print('现在r=', r)
                    gamma = np.random.uniform(0, 1 - r)
                    r += gamma
                    #print('之后r=', r)
                    k = 1
                    tauTime += 1
            if empty==1:
                w_vector_f = w_vector_temp  ## _f表示最终的star选择值
                b_f = b_temp
                break
            #else:
                #print('现在已经执行的lambda_=:',lambda_,'***接下来执行下一个lambda_')
            #####步骤4到16####尾

        #####步骤18到22####头
        # 训练
        featureMap = np.dot(x_tr, w_vector_f) + b_f
        # print 'x_tr.shape',x_tr.shape
        g = 1. / (1 + np.exp(-featureMap))

        beta.append(funcBeta(g, e_1))
        temp22 = np.dot(funcBeta(g, e_1), g)
        temp22 = temp22.reshape(len(temp22))
        e_t = e_1 - temp22
        yHat_tr += temp22
        e_0 = e_t
        # 测试
        featureMap_te = np.dot(x_te, w_vector_f) + b_f
        g_te = 1. / (1 + np.exp(-featureMap_te))
        temp33 = np.dot(funcBeta(g, e_1), g_te)
        temp33 = temp33.reshape(len(temp33))
        e_t_te = e_1_te - temp33
        yHat_te += temp33
        e_0_te = e_t_te
        #####步骤18到22####尾

        #print 'L=:', L, 'yHat_tr=:', yHat_tr, 'yHat_tr=:', yHat_tr
        L += 1
    Y_mean=np.ones_like(y_te)*np.mean(y_te)
    r2=1-(((yHat_te-y_te)**2).sum())/(((y_te-Y_mean)**2).sum())#
    n=len(yHat_te)
    err=y_te-yHat_te
    err_mean=np.ones_like(err)*np.mean(err)
    err_var=(((err-err_mean)**2).sum())/(n-1)
    y_var=(((y_te-Y_mean)**2).sum())/(n-1)
    Evar=1-err_var/y_var
    mse_tr = ((float(1) / len(y_tr)) * np.linalg.norm((y_tr - yHat_tr), ord=2) ** 2)
    mse_te = ((float(1) / len(y_te)) * np.linalg.norm((y_te - yHat_te), ord=2) ** 2)
    result.append([mse_tr,mse_te,r2,Evar])
    print(mse_tr,mse_te)
result=pd.DataFrame(result)
print(np.mean(result))
print(np.std(result))

# plt.figure(1)
# plt.scatter(x_tr, y_tr)
# plt.scatter(x_tr, yHat_tr)
# plt.show()
#print('这里是mse_tr:',mse_tr,'这里是mse_te',mse_te)
