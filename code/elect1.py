# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 20:41:38 2018

@author: Winry
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time,datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\simsun.ttc")


def secondsFrom1970(seq,name):
    temp = []
    newName = "New" + name
    for i in range(len(seq[name])):
        timeDateStr = seq[name][i]
        timeDateStr = datetime.datetime.strptime(timeDateStr, "%Y/%m/%d %H:%M")
        From1970 = time.mktime(timeDateStr.timetuple())
        temp.append(From1970)
    seq[newName] = temp

f = open('重要性确定.csv')
df = pd.read_csv(f)
f.close()



NameOne = 'plan_time'
secondsFrom1970(df,NameOne)
#df = df.iloc[1:8000,:]
df.rename(columns={'Newplan_time':'plantime'}, inplace = True)

data = df.iloc[:,1:16]
data = data.dropna(axis=0)
#trainy = data['time']
#data = data.drop(columns=['time'])
#trainx = data
y = np.round(data['time'])
temp = []
for i in range(len(data)):
    temp.append((data.iloc[i,-1] - data.iloc[0,-1])/(data.iloc[-1,-1] - data.iloc[0,-1]))
data.iloc[:,-1] = temp
x = data.drop(columns=['time'])

#x, y = data.iloc[:, 1:].values,data.iloc[:, 0].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
feat_labels = x.columns[0:]
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(x_train, y_train)
OriginalScore = forest.score(x_test,y_test)
temp_data = x

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


model = GradientBoostingClassifier()

model.fit(x_train, y_train)

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

#
#feat_lables = x_train.columns
#forest = RandomForestClassifier(n_estimators=100, random_state=0,n_jobs=1)
#forest.fit(trainx, trainy.astype('int'))
#importances = forest.feature_importances_
#imp_result = np.argsort(importances)[::-1]
#imp_result [0:5]
#indices = np.argsort(importances)[::-1]
temp_name = []
for f in range(x_train.shape[1]):
    temp_name.append(feat_labels[indices[f]])
    
plt.title('Feature Importance')
plt.bar(range(x_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(x_train.shape[1]), temp_name, rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.tight_layout()
plt.show()

temp_name.reverse()
temp_data = x_train
Score = []
for i in temp_name:
    if i == temp_name[-1]:
        break
    temp_data = temp_data.drop(columns=i)
    trainX, testX, trainY, testY = train_test_split(temp_data, y_train, test_size=0.3, random_state=0)
    model = GradientBoostingClassifier()
    model.fit(trainX, trainY)
    Score.append(model.score(testX, testY))
print(Score)
Score = [i + 0.15155682693365607 for i in Score]

kk = Score
Score = [i - 0.15155682693365607 for i in Score]


fig = plt.figure(figsize=(20,20),dpi=80)
plt.plot(range(len(Score)),Score)
plt.tick_params(labelsize=53)
#plt.title('Variable error diagram')
plt.xlabel('删除特征个数',fontproperties=my_font,size=53)
plt.ylabel('相对误差',fontproperties=my_font,size=53)
