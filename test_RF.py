# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 17:19:58 2021

@author: ohshi
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

smn1= 'SEVEN'
smn2= 'MARU'

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

# =============================================================================
# df1=df1[df1['distance']=='0.540']
# df2=df2[df2['distance']=='0.540']
# 
# =============================================================================
y = np.concatenate([df1['sample'],df2['sample']])

print(smn1,':', len(df1), ',', smn2, ':', len(df2))

#目的変数yの調整および作成
y = pd.get_dummies(y).drop(smn2,axis=1)
y=y.values


#説明変数Xの調整および作成
dfn1=df1[data]
dfn2=df2[data]

X = np.concatenate([dfn1,dfn2])

#テストデータと訓練データの分割
test_size=0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#データの不均衡の補正（アンダーサンプリング）
count0 = y_train[y_train==0].shape[0]
count1 = y_train[y_train==1].shape[0]

count = min(count0, count1)
strategy = {0:count, 1:count}

rus = RandomUnderSampler(random_state=0, sampling_strategy = strategy)

X_resampled, y_resampled = rus.fit_resample(X_train, y_train)


#ランダムフォレストを用いた学習
model =  RandomForestClassifier(random_state=0,n_estimators=100)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)

acc = f1_score(y_test, y_pred, average="micro")

mtx = confusion_matrix(y_test, y_pred)

MX = pd.DataFrame(mtx, index=['pred_'+smn1, 'pred_'+smn2], columns=['real_'+smn1,'read_'+smn2])

# 予測結果と、正解（本当の答え）がどのくらい合っていたかを表す混合行列の標準化
x_sm=(sum(mtx[0]))
y_sm=(sum(mtx[1]))
x_arry=mtx[0]/x_sm*100
y_arry=mtx[1]/y_sm*100

#混同行列（％）のデータとカラムラベル挿入
n_mtx=x_arry,y_arry
MX1=pd.DataFrame(n_mtx, index=['pred_'+smn1, 'pred_'+smn2], columns=['real_'+smn1,'read_'+smn2])


print(acc,MX)


# =============================================================================
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import cross_val_score
# 
# stratifiedkfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# print('Cross-validation scores: \n{}'.format(cross_val_score(model, X, y, cv=stratifiedkfold)))
# =============================================================================

importances = model.feature_importances_
indices = np.argsort(importances)

columns = [data[indices[i]] for i in indices]

import matplotlib.pyplot as plt

#values, names = zip(*sorted(zip(fimportances, features)))

plt.figure(figsize=(6,6))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), columns)
plt.show()

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

