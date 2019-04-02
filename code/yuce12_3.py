# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 21:19:02 2018

@author: Winry
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time,datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


label = ['plantime','meantime_taxi_number','mean_taxi_time','REG','distance','gate','land_takeoff_number',
         'destination','cat','airline','peak','runway','CLA']
label.reverse()

def secondsFrom1970(seq,name):
    temp = []
    newName = "New" + name
    for i in range(len(seq[name])):
        timeDateStr = seq[name][i]
        timeDateStr = datetime.datetime.strptime(timeDateStr, "%Y/%m/%d %H:%M")
        From1970 = time.mktime(timeDateStr.timetuple())
        temp.append(From1970)
    seq[newName] = temp

f = open('特征选择.csv')
df = pd.read_csv(f)
f.close()


df['a'] = df.iloc[:,0]
NameOne = 'a'
secondsFrom1970(df,NameOne)
df.rename(columns={'Newa':'plantime'}, inplace = True)
df = df.drop(columns=['a'])
data = df.iloc[:,1:15]
data = data.dropna(axis=0)
y = np.round(data['time'])
temp = []
for i in range(len(data)):
    temp.append((data.iloc[i,-1] - data.iloc[0,-1])/(data.iloc[-1,-1] - data.iloc[0,-1]))
data.iloc[:,-1] = temp
x = data.drop(columns=['time'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(x_train, y_train)
OriginalScore = forest.score(x_test,y_test)
temp_data = x

Score = []
for i in label:
    print(i)
    i == 'plantime'
    break
    temp_data = temp_data.drop(columns=i)
    x_train, x_test, y_train, y_test = train_test_split(temp_data, y, test_size=0.3, random_state=0)
    forest = RandomForestClassifier(n_estimators=100, random_state=0)
    forest.fit(x_train, y_train)
    Score.append(forest.score(x_test, y_test))
print(Score)