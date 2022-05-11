# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 14:33:46 2021

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

from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

smn1 = 'SEVEN'
smn2 = 'MEB'
smn3 = 'MARU'

sam1 = smn1+'_10k_Sample_ANAL_r'
sam2 = smn2+'_10k_Sample_ANAL_r'
sam3 = smn3+'_10k_Sample_ANAL_r'

datum1 = np.load('a_data/'+sam1+'.npy', allow_pickle=True)
datum2 = np.load('a_data/'+sam2+'.npy', allow_pickle=True)
datum3 = np.load('a_data/'+sam3+'.npy', allow_pickle=True)

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
df3 = pd.DataFrame(data=datum3, columns=columns)

y = np.concatenate([df1['sample'],df2['sample'],df3['sample']])

print(smn1,':', len(df1), ',', smn2, ':', len(df2), smn3,':', len(df3))

dfn1=df1[data]
dfn2=df2[data]
dfn3=df3[data]

X = np.concatenate([dfn1,dfn2,dfn3])

test_size=0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

count0 = y_train[y_train==smn1].shape[0]
count1 = y_train[y_train==smn2].shape[0]
count2 = y_train[y_train==smn3].shape[0]

count = min(count0, count1, count2)
strategy = {smn1:count, smn2:count, smn3:count}

rus = RandomUnderSampler(random_state=0, sampling_strategy = strategy)

X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

model =  RandomForestClassifier(random_state=0,n_estimators=100)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)

acc = f1_score(y_test, y_pred, average="micro")

mtx = confusion_matrix(y_test, y_pred)

MX = pd.DataFrame(mtx, index=['pred_'+smn1, 'pred_'+smn2, 'pred_'+smn3], columns=['real_'+smn1,'read_'+smn2,'read_'+smn3])

x_sm=(sum(mtx[0]))
y_sm=(sum(mtx[1]))
z_sm=(sum(mtx[2]))
x_arry=mtx[0]/x_sm*100
y_arry=mtx[1]/y_sm*100
z_arry=mtx[2]/z_sm*100

n_mtx=x_arry,y_arry,z_arry

MX1 = pd.DataFrame(n_mtx, index=['pred_'+smn1, 'pred_'+smn2,  'pred_'+smn3], columns=['real_'+smn1,'read_'+smn2,'read_'+smn3])

importances = model.feature_importances_
indices = np.argsort(importances)

columns = [data[indices[i]] for i in indices]

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
ax.set_title('Normalized_Confusion_Matrix')

print(acc)