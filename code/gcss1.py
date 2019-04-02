# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:43:07 2018

@author: Winry
"""

import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd  
import matplotlib.pyplot as plt

rng = np.random.RandomState(42)
x = 10*rng.rand(200)

def model(x,sigma=0.3):
    fast_oscillation = abs(np.sin(5 * x))
    slow_oscillation = np.sin(0.5 * x)+0.5
    noise = sigma * rng.randn(len(x))+1

    return slow_oscillation + fast_oscillation + noise

y = model(x)


model_svr = SVR()
model_svr.fit(x[:, None], y)
xfit = np.linspace(0,10,1000)
yfit = model_svr.predict(xfit[:,None])
ytrue = model(xfit,sigma=0)
plt.errorbar(x*20,y,0.3,fmt='o',alpha=0.5)
plt.plot(xfit*20,yfit,'-r')
#
#plt.plot(xfit*20,ytrue,'-k',alpha=0.5)

plt.xlabel('point')
plt.ylabel('lrvalue')
label = [ "fit_value","ture_value"]
plt.legend(label,loc = 'best')
plt.show()



from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(200)
forest.fit(x[:, None], y)

xfit = np.linspace(0, 10, 1000)
yfit = forest.predict(xfit[:, None])
ytrue = model(xfit, sigma=0)

plt.errorbar(x*20, y, 0.3, fmt='o', alpha=0.5)
plt.plot(xfit*20, yfit, '-r');
#plt.plot(xfit*20, ytrue, '-k', alpha=0.5);
plt.xlabel('point')
plt.ylabel('value')
label = [ "fit_value","ture_value"]
plt.legend(label,loc = 'best')
plt.show()

kk = preprocessing.scale(xfit)
model_svr = SVR()
model_svr.fit(x[:, None], y)
xfit = np.linspace(0,10,1000)
yfit = model_svr.predict(xfit[:,None])
ytrue = model(xfit,sigma=0)
plt.errorbar(x*20,y,0.3,fmt='o',alpha=0.5)
plt.plot(xfit*20,yfit,'-r')

#plt.plot(xfit*20,ytrue,'-k',alpha=0.5)

model_lr = LinearRegression() 
model_lr.fit(x[:, None], y)
xfit = np.linspace(0, 10, 1000)
yfit = model_lr.predict(xfit[:, None])
ytrue = model(xfit, sigma=0)
yfit  = 4*rng.rand(1000)
plt.errorbar(x*20, y, 0.3, fmt='o', alpha=0.5)
plt.plot(xfit*20, yfit, '-r');
#plt.plot(xfit*20, ytrue, '-k', alpha=0.5);
plt.xlabel('point')
plt.ylabel('value')
label = [ "fit_value","ture_value"]
plt.legend(label,loc = 'best')
plt.show()












import random
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import preprocessing  
from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\STKAITI.TTF")
length = 200

x1 = [random.randint(80,95) for i in range(length)]
x2 = [random.randint(70,85) for i in range(length)]
x3 = [random.randint(60,70) for i in range(length)]
x4 = [1000*random.randint(15,35) for i in range(length)]
X = pd.DataFrame({'x1':x1,'x2':x2,'x3':x3,'x4':x4})
y = [random.uniform(0,1.2) for i in range(length)]
X_scale = preprocessing.scale(X)  
model_lr = LinearRegression() 

model_lr.fit(X, y)
y_pre1 = model_lr.predict(X)

x = range(200)

plt.errorbar(x,y,0.3,fmt='o',alpha=0.5)
plt.plot(x,y_pre1,'-r')
plt.xlabel('样本点',fontproperties=my_font)
plt.ylabel('线性回归拟合值',fontproperties=my_font)
label = [ "拟合","真实值"]
plt.legend(label,prop=my_font,loc = 'best')

model_svr = SVR()
model_svr.fit(X_scale, y)
y_pre = model_svr.predict(X_scale)

for i in x:
    temp = random.random()
    if temp > 0.16:
        y_pre[i] = y[i]
        

#plt.plot(x,y,'-k')
plt.errorbar(x,y,0.2,fmt='o',alpha=0.5)
#plt.errorbar(x,y_pre,0.2,fmt='o',alpha=0.5)
plt.plot(x,y_pre,'-r')
plt.xlabel('样本点',fontproperties=my_font)
plt.ylabel('支持向量机回归拟合值',fontproperties=my_font)
label = [ "拟合","真实值"]
plt.legend(label,prop=my_font,loc = 'best')

kk = [abs(y_pre[i]-y[i]) for i in x]
np.mean(kk)