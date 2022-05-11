# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:11:20 2021

@author: ohshi
"""
#モジュール読み出し
import numpy as np
import keras
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#モジュール読み出し
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras.layers import Dense, Dropout

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler


#サンプル名リスト
smns=['pSer','LSer','pThr','LThr']#'AAT','ACT','AGT','ATT','CA','CC','CG','CT','GA','GC','GG','GT','TA','TC','TG','TT'

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

#混合行列用カラムとインデックス作成    
mtx_index = ['pred_'+smn for smn in smns]
mtx_columns = ['real_'+smn for smn in smns]

#目標変数(y)の配列作成
y = [_.split('_')[0] for _ in dnf['sample']]

#説明変数(X)の配列作成
x = dnf[data]
#説明変数(X)のデータの標準化
X = preprocessing.scale(x)

#目標変数(y)の文字データの整数への変換
le = LabelEncoder()
#ラベルを覚えさせる
le = le.fit(y)
#ラベルを整数に変換
y = le.transform(y)

#目標変数(y)のクラス数
num_class = max(y)+1

#データ分割
test_size=0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)


#目標変数の偏りの是正(Undersampling)
cn=[len(y_train[y_train==i]) for i in range(num_class)]

counts = [min(cn) for _ in range(len(cn))]
keys = [_ for _ in range(len(cn))]

strategy = {key:count for key, count in zip(keys,counts)}

rus = RandomUnderSampler(random_state=0, sampling_strategy = strategy)

X_train, y_train = rus.fit_resample(X_train, y_train)

# 訓練データとテストデータのshapeをチェック
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#(398, 30) (171, 30) (398, 2) (171, 2)

# モデル構築
def build_model():
    model = Sequential()
    model.add(Dense(32,activation='relu',input_shape=(30,)))
    model.add(Dropout(0.2))
    model.add(Dense(16,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_class,activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

#K-Fold検証法(https://manareki.com/k_fold)
kf = KFold(n_splits=3, shuffle=True)#n_splitsは分割回数(３分割)
all_loss=[]
all_val_loss=[]
all_acc=[]
all_val_acc=[]
ep=100

for train_index, val_index in kf.split(X_train,y_train):

    train_data=X_train[train_index]
    train_label=y_train[train_index]
    val_data=X_train[val_index]
    val_label=y_train[val_index]

    model=build_model()
    history=model.fit(train_data,
                      train_label,
                      epochs=ep,
                      batch_size=8,
                      validation_data=(val_data,val_label))

    loss=history.history['loss']
    val_loss=history.history['val_loss']
    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']

    all_loss.append(loss)
    all_val_loss.append(val_loss)
    all_acc.append(acc)
    all_val_acc.append(val_acc)
    
ave_all_loss=[
    np.mean([x[i] for x in all_loss]) for i in range(ep)]
ave_all_val_loss=[
    np.mean([x[i] for x in all_val_loss]) for i in range(ep)]
ave_all_acc=[
    np.mean([x[i] for x in all_acc]) for i in range(ep)]
ave_all_val_acc=[
    np.mean([x[i] for x in all_val_acc]) for i in range(ep)]


#指標の履歴表示（loss）    
plt.plot(history.history['loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

#指標の履歴表示（accuracy）
plt.plot(history.history['accuracy'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

#モデルをもちいた予想
predict_classes = model.predict_classes(X_test, batch_size=32)
true_classes = np.argmax(y_test,1)
matx=confusion_matrix(true_classes, predict_classes)

# 予測結果と、正解（本当の答え）がどのくらい合っていたかを表す混合行列の標準化
x_sm=(sum(matx[0]))
y_sm=(sum(matx[1]))
x_arry=matx[0]/x_sm
y_arry=matx[1]/y_sm

#混同行列（％）のデータとカラムラベル挿入
n_matx=x_arry,y_arry
MX2=pd.DataFrame(n_matx, index=[u'predict_Non-cancer',
                          u'predict_Cancer'], columns=[u'read_Non Cancer', u'real_Cancer'])

#混同行列（数）のデータとカラムラベル挿入
MX1=pd.DataFrame(matx, index=[u'predict_Non-cancer',
                          u'predict_Cancer'], columns=[u'read_Non Cancer', u'real_Cancer'])

#モデルをもちいた予想の可視化
fig, ax = plt.subplots(figsize=(4, 3)) # 混合行列のカラムの大きさ設定
sns.heatmap(MX1, annot=True, fmt="d")
ax.set_ylim(len(matx), 0)# 混合行列の軸の下限を設定し，値がみえるようにする（バグ）
ax.set_title('Confusion_Matrix')

#モデルをもちいた予想の可視化
fig, ax = plt.subplots(figsize=(4, 3)) # 混合行列のカラムの大きさ設定
sns.heatmap(MX2, annot=True, fmt="1.3")# fmtでデータの表示桁数
ax.set_ylim(len(matx), 0)# 混合行列の軸の下限を設定し，値がみえるようにする（バグ）
ax.set_title('normalized confusion matrix')