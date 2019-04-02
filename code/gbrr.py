# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:05:09 2018

@author: Winry
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time,datetime
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法


def secondsFrom1970(seq,name):
    temp = []
    newName = "New" + name
    for i in range(len(seq[name])):
        timeDateStr = seq[name][i]
        timeDateStr = datetime.datetime.strptime(timeDateStr, "%Y/%m/%d %H:%M")
        From1970 = time.mktime(timeDateStr.timetuple())
        temp.append(From1970)
    seq[newName] = temp
    
f = open('xunlian.csv')
df1 = pd.read_csv(f)
f.close()

f = open('yuce.csv')
df2 = pd.read_csv(f)
f.close()

df1['a'] = df1.iloc[:,0]
NameOne = 'a'
secondsFrom1970(df1,NameOne)
df1.rename(columns={'Newa':'plantime'}, inplace = True)
df1 = df1.drop(columns=['a'])
data = df1.iloc[:,1:15]
data = data.dropna(axis=0)
y_train = (data['time'])
temp = []
for i in range(len(data)):
    temp.append((data.iloc[i,-1] - data.iloc[0,-1])/(data.iloc[-1,-1] - data.iloc[0,-1]))
data.iloc[:,-1] = temp
X_train = data.drop(columns=['time'])


df2['a'] = df2.iloc[:,0]
NameOne = 'a'
secondsFrom1970(df2,NameOne)
df2.rename(columns={'Newa':'plantime'}, inplace = True)
df2 = df2.drop(columns=['a'])
data = df2.iloc[:,1:15]
data = data.dropna(axis=0)
y_test = (data['time'])
temp = []
for i in range(len(data)):
    temp.append((data.iloc[i,-1] - data.iloc[0,-1])/(data.iloc[-1,-1] - data.iloc[0,-1]))
data.iloc[:,-1] = temp
X_test = data.drop(columns=['time'])


# 训练回归模型
n_folds = 6  # 设置交叉检验的次数
model_br = BayesianRidge() 
model_lr = LinearRegression() 
model_etc = ElasticNet() 
model_svr = SVR() 
model_gbr = GradientBoostingRegressor() 
model_names = ['BayesianRidge', 'LinearRegression', 'ElasticNet', 'SVR', 'GBR'] 
model_dic = [model_br, model_lr, model_etc, model_svr, model_gbr] 
cv_score_list = []  # 交叉检验结果列表
pre_y_list = []  # 各个回归模型预测的y值列表
for model in model_dic:  # 读出每个回归模型对象
    scores = cross_val_score(model, X_train, y_train, cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验
    cv_score_list.append(scores)  # 将交叉检验结果存入结果列表
    pre_y_list.append(model.fit(X_train, y_train).predict(X_train))  # 将回归训练中得到的预测y存入列表
# 模型效果指标评估
n_samples, n_features = X_train.shape  # 总样本量,总特征数
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表
for i in range(5):  # 循环每个模型索引
    tmp_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(y_train, pre_y_list[i])  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
    model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框
df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框
print ('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量
print (70 * '-')
print ('cross validation result:')  # 打印输出标题
print (df1)  # 打印输出交叉检验的数据框
print (70 * '-') 
print ('regression metrics:')  # 打印输出标题
print (df2)  # 打印输出回归指标的数据框
print (70 * '-')
print ('short name \t full name')  # 打印输出缩写和全名标题
print ('ev \t explained_variance')
print ('mae \t mean_absolute_error')
print ('mse \t mean_squared_error')
print ('r2 \t r2')
print (70 * '-')  # 打印分隔线
# 模型效果可视化
plt.figure()  # 创建画布
plt.plot(np.arange(X_train.shape[0]), y_train, color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'c']
linestyle_list = ['-', '.', 'o', 'v', '*']
for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(np.arange(X_train.shape[0]), pre_y_list[i], color_list[i], label=model_names[i])  # 画出每条预测结果线
plt.title('regression result comparison')
plt.legend(loc='upper right')
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
print ('regression prediction')
cc = []

for i in range(len(X_test)):
    new_point = X_test.iloc[i,:]
    new_point = new_point.values
    new_point = np.array(new_point).reshape(1,-1)
    new_pre_y = model_gbr.predict(new_point)
    cc.append(list(np.around(new_pre_y))[0])
gbr_count = 0


def model_count(X_test,y_test,model,num):
    cc = []
    counts = 0
    for i in range(len(X_test)):
        new_point = X_test.iloc[i,:]
        new_point = new_point.values
        new_point = np.array(new_point).reshape(1,-1)
        new_pre_y = model.predict(new_point)
        cc.append(list(np.around(new_pre_y))[0])
    for i in range(len(y_test)):
        if (cc[i] <= y_test.iloc[i]+num)&(cc[i] >= y_test.iloc[i]-num):
            counts += 1
    return cc,counts

[gbr_cc,gbr_count] = model_count(X_test,y_test,model_gbr,0)
[br_cc,br_count] = model_count(X_test,y_test,model_br,3)
[lr_cc,lr_count] = model_count(X_test,y_test,model_lr,3)
[svr_cc,svr_count] = model_count(X_test,y_test,model_svr,3)
[etc_cc,etc_count] = model_count(X_test,y_test,model_etc,3)

gbr_residuals = y_test - gbr_cc
svr_residuals = y_test - svr_cc
lr_residuals = y_test - lr_cc
br_residuals = y_test - br_cc
etc_residuals = y_test - etc_cc

f, ax = plt.subplots(figsize=(7, 5))
f.tight_layout()
ax.hist(gbr_residuals,bins=20,label='GBR Residuals', color='b', alpha=.5);
#ax.hist(svr_residuals,bins=20,label='SVR Residuals', color='r', alpha=.5);
#ax.hist(lr_residuals,bins=20,label='lr Residuals', color='r', alpha=.5);
ax.hist(etc_residuals,bins=20,label='etc Residuals', color='r', alpha=.5);
#ax.hist(br_residuals,bins=20,label='br Residuals', color='r', alpha=.5);
ax.set_title("GBR Residuals vs etc Residuals")
ax.legend(loc='best');
plt.xlabel('errors')
plt.ylabel('Number of errors')

from sklearn.neighbors import KernelDensity

# instantiate and fit the KDE model
#kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
#kde.fit(gbr_residuals[:, None])
#x_d = np.linspace(-6, 10, 1000)
## score_samples returns the log of the probability density
#logprob = kde.score_samples(x_d[:, None])
#plt.fill_between(x_d, np.exp(logprob),  alpha=0.5)
#kde.fit(br_residuals[:, None])
#logprob = kde.score_samples(x_d[:, None])
#plt.fill_between(x_d, np.exp(logprob),  alpha=0.5)



plt(X_test['a'],y_test - gbr_cc)

plt.figure(figsize=(7,5),dpi=80)
zeross = np.zeros(622)
plt.plot(range(len(y_test)),y_test - gbr_cc,marker='o')
plt.plot(range(len(y_test)),zeross,'k')



