# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:33:01 2018

@author: Winry
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:36:20 2018

@author: Winry
"""

import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd  
import matplotlib.pyplot as plt


#df_train = pd.read_csv('train.csv')
#df_test = pd.read_csv('test.csv')
#
#df_train = df_train.dropna(axis=0)
#df_test = df_test.dropna(axis=0)
#
#X_train = df_train.iloc[:,0:6]
#y_train = df_train.iloc[:,-2]
#X_train['ytrain'] =y_train
#X_train = X_train.dropna(axis=0)
#y_train = X_train['ytrain']
#X_train = X_train.iloc[:,0:10]
#
#X_test = df_test.iloc[0:150,1:11]
#y_test = df_test.iloc[0:150,-1]
#
#X_test = X_test.dropna(axis=0)
#y_test = y_test.dropna(axis=0)

#X_train = preprocessing.scale(X_train)
##y_train = preprocessing.scale(y_train)
#X_test = preprocessing.scale(X_test)
#y_test = preprocessing.scale(y_test)

# 数据准备
df = pd.read_csv('d11_13.csv')
df = df.iloc[:,0:10]
df = df.dropna(axis=0)

X = df.iloc[:,0:7]
y = df['forecast']

label_train = []
label_test = []
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=15)
for i in range(len(X)):
    temp = np.matlib.rand(1)
    if temp >= 0.8:
        label_test.append(i)
    else:
        label_train.append(i)
X_train = X.iloc[label_train,:]
y_train = y.iloc[label_train]


df_train = pd.read_csv('train.csv')
X_train =df_train.dropna(axis=0).iloc[:,0:7]
y_train = df_train.dropna(axis=0).iloc[:,-2]


df_test = pd.read_csv('test.csv')


df_test = df_test.dropna(axis=0)
X_test = df_test.iloc[:,0:7]
y_test = df_test.iloc[:,-2]





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
    cc.append(list(np.around(new_pre_y)))
gbr_count = 0
for i in range(len(y_test)):
    if (cc[i][0] <= y_test.iloc[i]+2)&(cc[i][0] >= y_test.iloc[i]-2):
#    if (cc[i][0] == y_test.iloc[i]):
        gbr_count += 1
k11=[]
for i in range(len(cc)):
    k11.append(cc[i][0])

        

