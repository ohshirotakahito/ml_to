# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 13:23:22 2021

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
import amin_abb as ab

#混合アミノ酸サンプルフォルダ
file = '2G_L_T_D_WMN_10k_Sample_ANAL_r'

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
  

#混合サンプルのファイルのロード
datum = np.load('a_data/'+file+'.npy',allow_pickle=True)
 
dft = pd.DataFrame(data=datum, columns=columns)
x_t = dft[data]
x_t = preprocessing.scale(x_t)
 
dtest_t = xgb.DMatrix(x_t)

model_xgb = xgb.Booster()
bst = model_xgb.load_model("model.json")
y_pred_t = bst.predict(dtest_t)
 #d_labels = le.inverse_transform(y_pred_t)
y_pred_t = y_pred_t.astype(int)

d_t = le.inverse_transform(y_pred_t)
 
amin_c=[]
for i in range(len(smns)):
     ci = np.count_nonzero(d_t == smns[i])
     k = smns[i]
     p_ci = '{:.3g}'.format(ci/len(x_t)*100)
     print(f'{k} = {ci}')
     print(f'{k} (%) = {p_ci}')
     amin_c.append(ci)
 
spns = []
for sm in smns:
     if 'D' in sm:
         spns.append(0)
     elif 'L' in sm:
         spns.append(1)
     else:
         spns.append(2)
 
Dn = 0
Ln = 0
Gn = 0
 
for i in spns:
     p = amin_c[i]
     if i==0:
         Dn = Dn + p
     elif i ==1:
         Ln = Ln + p
     else:
         Gn = Gn + p
 
Dp = '{:.3g}'.format(Dn/len(x_t)*100)
Lp = '{:.3g}'.format(Ln/len(x_t)*100)
Gp = '{:.3g}'.format(Gn/len(x_t)*100)
 
print(f'D = {Dn}  D (%) = {Dp}')
print(f'L = {Ln}  L (%) = {Lp}')
print(f'G = {Gn}  G (%) = {Gp}')
    

if __name__ =='__main__':
    #混合アミノ酸サンプルフォルダ
    files = ['2G_L_T_D_WMN_10k_Sample_ANAL_r',
             ]
    file = '2G_L_T_D_WMN_10k_Sample_ANAL_r' 
    mixedamin(file)