# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 17:42:38 2018

@author: Winry
"""

from sklearn.ensemble import GradientBoostingRegressor as GBR

import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd  
import matplotlib.pyplot as plt
gbr = GBR()
gbr.fit(X, y)

from sklearn.ensemble import GradientBoostingRegressor as GBR
gbr = GBR()
gbr.fit(X_train, y_train)
gbr_preds = np.around(gbr.predict(X_test))


from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=200)
clf.fit(X_train, y_train)
clf_preds = clf.predict(X_test)



count1 = 0
for i in range(len(clf_preds)):
    if (clf_preds[i] <= y_test.iloc[i]+2)&(clf_preds[i] >= y_test.iloc[i]-2):
        count1 += 1