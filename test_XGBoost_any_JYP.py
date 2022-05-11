# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 12:00:39 2021

@author: ohshi
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

import xgboost as xgb

smns=['D','LSer','LAla','LThr','pThr']

#['Ala':'A','Arg':'R',

#元ファイルのカラムリスト
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

#訓練で用いるカラムリスト
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

#データ読み込み
dnf = pd.DataFrame(columns=columns)
for smn in smns:
    sam = smn + '_10k_Sample_ANAL_r'
    datum = np.load('a_data/'+sam+'.npy',allow_pickle=True)
    df = pd.DataFrame(data=datum, columns=columns)
    dnf=pd.concat([dnf,df],axis=0)
    print(smn,':', len(df))
    
#目標変数(y)の配列作成
y = [_.split('_')[0] for _ in dnf['sample']]
#目標変数(y)の文字データの整数への変換
le = LabelEncoder()
#ラベルを覚えさせる
le = le.fit(y)
#ラベルを整数に変換
y = le.transform(y)
#目標変数(y)のクラス数
num_class = max(y)+1

#混合行列用カラムとインデックス作成 
smns_x=le.classes_
mtx_index = ['pred_'+smn_x for smn_x in smns_x]
mtx_columns = ['real_'+smn_x for smn_x in smns_x]

#説明変数(X)の配列作成
x = dnf[data]
#説明変数(X)のデータの標準化
X = preprocessing.scale(x)

#データ分割
test_size=0.8

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)


#目標変数の偏りの是正(Undersampling)
cn=[len(y_train[y_train==i]) for i in range(num_class)]

counts = [min(cn) for _ in range(len(cn))]
keys = [_ for _ in range(len(cn))]

strategy = {key:count for key, count in zip(keys,counts)}

rus = RandomUnderSampler(random_state=0, sampling_strategy = strategy)

X_train_re, y_train_re = rus.fit_resample(X_train, y_train)

#訓練用データをxgb.DMatrixで，XGBoost用のデータ型に変換
dtrain = xgb.DMatrix(X_train_re, label=y_train_re)
dtest = xgb.DMatrix(X_test)

params = {'max_depth': 10, 
          'eta': 1, 
          'objective': 'multi:softmax', 
          'num_class': num_class}
num_round = 200

bst = xgb.train(params, dtrain, num_round)

y_pred = bst.predict(dtest)

acc = f1_score(y_test, y_pred, average="micro")

mtx = confusion_matrix(y_test, y_pred)
MX = pd.DataFrame(mtx, index=mtx_index, columns=mtx_columns)

n_mtx=[mtx[i]/(sum(mtx[i]))*100 for i in range(len(mtx))]
N_MX = pd.DataFrame(n_mtx, index=mtx_index, columns=mtx_columns)

report = classification_report(y_test, y_pred)

print('f-measure_value:', acc)

#モデルをもちいた予想の可視化
fig, ax = plt.subplots(figsize=(4, 3)) # 混合行列のカラムの大きさ設定
sns.heatmap(MX, annot=True, fmt="d",center=250)
ax.set_ylim(len(mtx), 0)# 混合行列の軸の下限を設定し，値がみえるようにする（バグ）
ax.set_title('Confusion_Matrix')

#モデルをもちいた予想の標準化後の可視化
fig, ax = plt.subplots(figsize=(4, 3)) # 混合行列のカラムの大きさ設定
sns.heatmap(N_MX, annot=True, fmt="1.1f",center=250)
ax.set_ylim(len(n_mtx), 0)# 混合行列の軸の下限を設定し，値がみえるようにする（バグ）
ax.set_title('Normalized_Confusion_Matrix')