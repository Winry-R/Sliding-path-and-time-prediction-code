# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 17:14:14 2018

@author: Winry
"""

import pandas as pd
# 显示所有的列
pd.set_option('display.max_columns', None)

# 读取数据
file_name = "data_11_8.csv"
file_open = open(file_name)
df = pd.read_csv(file_open)
file_open.close()

Newtaxiout_time = df['Newtaxiout_time']
time = df['time']

file_name2 = "df_append.csv"
file_open2 = open(file_name2)
df2 = pd.read_csv(file_open2)

# append1

append1_res = []
for i in range(len(df)):
    count = []
    count = df2["Newappend1"][(df2["Newappend1"] > Newtaxiout_time[i]) & (df2["Newappend1"] < time[i]*60+Newtaxiout_time[i])]
    append1_res.append(len(count))


# append2
append2_res = []
for i in range(len(df)):
    count = []
    count = df2["Newappend2"][(df2["Newappend2"] > Newtaxiout_time[i]) & (df2["Newappend2"] < time[i]*60+Newtaxiout_time[i])]
    append2_res.append(len(count))
    
df['append1_res'] = append1_res
df['append2_res'] = append2_res
df.to_csv('df_11_9.csv',index=False)