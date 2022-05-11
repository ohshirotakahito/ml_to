# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 13:32:27 2021

@author: ohshi
"""
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

datum1 = np.load(sam1+'.npy', allow_pickle=True)
datum2 = np.load(sam2+'.npy', allow_pickle=True)

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



#トレーニングデータ、テストデータの分離
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#グリッドサーチ
for model, param in tqdm(SVC_grid.items()):
    clf = GridSearchCV(model, param)
    clf.fit(train_X, train_y)
    pred_y = clf.predict(test_X)
    score = f1_score(test_y, pred_y, average="micro")

    if max_score < score:
        max_score = score
        best_param = clf.best_params_
        best_model = model.__class__.__name__

print("ベストスコア:{}".format(max_score))
print("モデル:{}".format(best_model))
print("パラメーター:{}".format(best_param))

#ハイパーパラメータを調整しない場合との比較
model = SVC()
model.fit(train_X, train_y)
score = model.score(test_X, test_y)
print("")
print("デフォルトスコア:", score)