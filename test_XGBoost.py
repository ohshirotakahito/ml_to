# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:16:06 2021

@author: ohshi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import preprocessing

import xgboost as xgb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from imblearn.under_sampling import RandomUnderSampler



smn1 = 'SEVEN'
smn2 = 'MARU'

sam1 = smn1+'_10k_Sample_ANAL_r'
sam2 = smn2+'_10k_Sample_ANAL_r'

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

print(smn1,':', len(df1), ',', smn2, ':', len(df2))

#説明変数Xの調整および作成
dfn1=df1[data]
dfn2=df2[data]

X = np.concatenate([dfn1,dfn2])

# データの標準化
X = preprocessing.scale(X)

#DOPCを1,　PSOCを0
y = pd.get_dummies(y).drop(smn2,axis=1).values

#訓練データと検証データの分離
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#データの不均衡の補正（アンダーサンプリング）
count0 = y_train[y_train==0].shape[0]
count1 = y_train[y_train==1].shape[0]

count = min(count0, count1)
strategy = {0:count, 1:count}

rus = RandomUnderSampler(random_state=0, sampling_strategy = strategy)

X_train, y_train = rus.fit_resample(X_train, y_train)



#訓練用データをxgb.DMatrixで，XGBoost用のデータ型に変換
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

#学習パラメータ設定
params = {'max_depth': 10, 
          'eta': 1, 
          'objective': 'multi:softmax', 
          'num_class': 3}
# =============================================================================
# params = {'max_depth': 4, 
#           'eta': 0.3, 
#         'objective': 'binary:logistic',
#         'silent':1, 
#         'random_state':1234, 
#         'eval_metric': 'rmse',
#     }
# =============================================================================

num_round = 200

#学習
bst = xgb.train(params, dtrain, num_round)

#予想
y_pred = bst.predict(dtest)

#評価
acc = accuracy_score(y_test, y_pred)

mtx = confusion_matrix(y_test, y_pred)
MX = pd.DataFrame(mtx, index=['pred_'+smn1, 'pred_'+smn2,], columns=['real_'+smn1,'read_'+smn2])

# 予測結果と、正解（本当の答え）がどのくらい合っていたかを表す混合行列の標準化
x_sm=(sum(mtx[0]))
y_sm=(sum(mtx[1]))
x_arry=mtx[0]/x_sm*100
y_arry=mtx[1]/y_sm*100

#混同行列（％）のデータとカラムラベル挿入
n_mtx=x_arry,y_arry

MX1=pd.DataFrame(n_mtx, index=['pred_'+smn1, 'pred_'+smn2], columns=['real_'+smn1,'read_'+smn2])

f = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print('f-measure_value:', f)
#print(MX)

#変数重要度を出力
mapper = {'f{0}'.format(i): v for i, v in enumerate(data)}
mapped = {mapper[k]: v for k, v in bst.get_fscore().items()}
xgb.plot_importance(mapped)


#モデルをもちいた予想の可視化
fig, ax = plt.subplots(figsize=(4, 3)) # 混合行列のカラムの大きさ設定
sns.heatmap(MX, annot=True, fmt="d",center=250)
ax.set_ylim(len(mtx), 0)# 混合行列の軸の下限を設定し，値がみえるようにする（バグ）
ax.set_title('Confusion_Matrix')

#モデルをもちいた予想の標準化後の可視化
fig, ax = plt.subplots(figsize=(4, 3)) # 混合行列のカラムの大きさ設定
sns.heatmap(MX1, annot=True, fmt="1.1f",center=250)
ax.set_ylim(len(n_mtx), 0)# 混合行列の軸の下限を設定し，値がみえるようにする（バグ）
ax.set_title('normalized Confusion_Matrix')

