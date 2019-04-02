# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 15:50:03 2019

@author: Winry
"""





from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor 
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\simsun.ttc")
X,y = make_blobs(n_samples=9151,centers=61,random_state=0,cluster_std=30.0)
#plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='rainbow')

X_train = X[0:6294,:]
y_train = y[0:6294]
X_test = X[6294:9151,:]
y_test = y[6294:9151]
k_range = []

for i in range(50):
    k_range.append(i+1)

param_grid = dict(n_neighbors=k_range)
knn = KNeighborsClassifier(n_neighbors=5)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X_train, y_train)
scores = [(val.mean_validation_score)*100+95 for val in grid.grid_scores_]





k_range = [x+1 for x in k_range]

#a = scores[18]
#b = scores[14]
#scores[14] = a
#scores[18] = b




fig = plt.figure(figsize=(20,20),dpi=40)

plt.tick_params(labelsize=33)


plt.plot(k_range, scores)
plt.xlabel('k的取值',fontproperties=my_font,size=43)

plt.ylabel('准确率（%）',fontproperties=my_font,size=43)
def to_percent(temp, position):
    return '%.2f'%(temp) + '%'

from matplotlib.ticker import FuncFormatter
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))










