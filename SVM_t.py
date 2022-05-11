# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:54:06 2021

@author: ohshi
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix

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

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = SVC(gamma='scale')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = f1_score(y_test, y_pred, average="micro")
mtx = confusion_matrix(y_test, y_pred)
MX = pd.DataFrame(mtx, index=['pred_'+smn1, 'pred_'+smn2,], columns=['real_'+smn1,'read_'+smn2])

print(acc,MX)