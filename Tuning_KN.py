# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 17:19:58 2021

@author: ohshi
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


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

param_gs_knn ={'est__n_neighbors':[1, 3, 5, 7, 9, 11, 15, 21],
               'est__weights':['uniform','distance'],
               'est__p':[1,2]}


best_score = 0
best_params = {}
for n_neighbors in tqdm(param_gs_knn['est__n_neighbors']):
    for weights in param_gs_knn['est__weights']:
        for p in param_gs_knn['est__p']:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors,
                                       weights=weights,p=p)
            # 交差検証によるハイパーパラメータの探索
            scores = cross_val_score(knn, X_train, y_train, cv=5)
            score = np.mean(scores)
            if score > best_score:
                best_score = score
                best_params = {'n_neighbors': n_neighbors,
                               'weights': weights,'p': p}


print(best_score)
print(best_params)
