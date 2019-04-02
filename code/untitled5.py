# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:29:23 2018

@author: Winry
"""

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
        
filename = 'sonar.all-data.csv'
dataset = load_csv(filename)

def str_column_to_float(dataset, column):
    for row in dataset:
        print(row[3])
        print("==========================")
        print(row[3].strip())
        row[column] = row[3].strip()
        print("==========================")
        print(row[3])
 