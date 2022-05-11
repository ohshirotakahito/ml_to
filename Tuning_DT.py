# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:33:12 2021

@author: ohshi
"""
# -*- coding: utf-8 -*-

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

sam1 = 'DSPC_10k_Sample_ANAL_r'
sam2 = 'DSPE_10k_Sample_ANAL_r'

smn1= sam1[:sam1.find('_10k_Sample_ANAL_r')]
smn2= sam2[:sam2.find('_10k_Sample_ANAL_r')]

datum1 = np.load('a_data/'+sam1+'.npy', allow_pickle=True)
datum2 = np.load('a_data/'+sam2+'.npy', allow_pickle=True)

columns=['file',
         'distance',
         'sample', 
         'sinal_position',
         'signal_intensity',
         'signal_time', 
         'signal_start',
         'signal_end',
        'sinal_baseline',
        'f1',
        'f2',
        'f3',
        'f4',
        'f5',
        'f6',
        'f7',
        'f8',
        'f9',
        'f10',
        'f11',
        'f12']

data=['signal_intensity',
         'signal_time',
        'sinal_baseline',
        'f1',
        'f2',
        'f3',
        'f4',
        'f5',
        'f6',
        'f7',
        'f8',
        'f9',
        'f10',
        'f11',
        'f12']

df1 = pd.DataFrame(data=datum1, columns=columns)
df2 = pd.DataFrame(data=datum2, columns=columns)

y = np.concatenate([df1['sample'],df2['sample']])


dfn1=df1[data]
dfn2=df2[data]

X = np.concatenate([dfn1,dfn2])

#DOPCを1,　PSOCを0
y = pd.get_dummies(y).drop(smn2,axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#条件設定
max_score = 0
SearchMethod = 0
DTC_grid = {DecisionTreeClassifier(): {"criterion": ["gini", "entropy"],
                                       "splitter": ["best", "random"],
                                       "max_depth": [1,2,4,6,8,10],
                                       "min_samples_split": [3,5,8,10],
                                       "min_samples_leaf": [1,2,5,8,10],
                                       "random_state": [1,10,20,40,60,80,100]
                                      }}

#決定木の実行
for model, param in tqdm(DTC_grid.items()):
    clf = GridSearchCV(model, param)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred, average="micro")

    if max_score < score:
        max_score = score
        best_param = clf.best_params_
        best_model = model.__class__.__name__

print("ベストスコア:{}".format(max_score))
print("モデル:{}".format(best_model))
print("パラメーター:{}".format(best_param))