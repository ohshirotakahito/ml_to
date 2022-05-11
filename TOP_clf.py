# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 17:19:58 2021

@author: ohshi
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


sam1 = 'DOPC_10k_Sample_ANAL_r'
sam2 = 'DSPC_10k_Sample_ANAL_r'

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = KNeighborsClassifier(n_neighbors=21, weights='distance',p=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)

mtx = confusion_matrix(y_test, y_pred)

MX = pd.DataFrame(mtx, index=['pred_'+smn1, 'pred_'+smn2], columns=['real_'+smn1,'read_'+smn2])

print(acc,MX)

# =============================================================================
# model2 = SVC(gamma='scale')
# model2.fit(X_train, y_train)
# y2_pred = model2.predict(X_test)
# 
# acc2 = metrics.accuracy_score(y_test, y2_pred)
# 
# mtx1 = confusion_matrix(y_test, y_pred)
# MX1 = pd.DataFrame(mtx1, index=['pred_DOPC', 'pred_DSPC',], columns=['real_DOPC','read_DSPC'])
# 
# print(acc,MX)
# =============================================================================

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

stratifiedkfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
print('Cross-validation scores: \n{}'.format(cross_val_score(model, X, y, cv=stratifiedkfold)))
