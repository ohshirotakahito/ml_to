# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 11:19:48 2021

@author: ohshi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.model_selection import KFold


from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras.layers import Dense, Dropout

from imblearn.under_sampling import RandomUnderSampler

smn1= 'LIN'
smn2= 'STE'

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

#目的変数yの調整および作成
#DOPCを1,　PSOCを0
y = pd.get_dummies(y).drop(smn2,axis=1).values
y = np.reshape(y,-1)

# ラベルをone-hot-encoding形式に変換
y = np_utils.to_categorical(y)

X = X.astype(float)

# 訓練データとテストデータに分割
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2)

# =============================================================================
# #データの不均衡の補正（アンダーサンプリング）
# count0 = y_train[y_train==0].shape[0]
# count1 = y_train[y_train==1].shape[0]
# 
# count = min(count0, count1)
# strategy = {0:count, 1:count}
# 
# rus = RandomUnderSampler(random_state=0, sampling_strategy = strategy)
# 
# X_train, y_train = rus.fit_resample(X_train, y_train)
# =============================================================================


def build_model():
    model = Sequential()
    model.add(Dense(16,input_shape=(15, )))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    model.compile(optimizer='SGD',##学習の最適化法を決定(勾配法)
                      loss='binary_crossentropy',#損失関数(誤差関数)を決定
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

    train_data= X_train[train_index]
    train_label= y_train[train_index]
    val_data= X_train[val_index]
    val_label= y_train[val_index]

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
y_pred = model.predict(X_test)


#目的関数の調整
y_test = np.argmax(y_test,1)
y_pred = np.argmax(y_pred,1)

#学習の評価結果
acc = f1_score(y_test, y_pred, average="micro")

#評価結果
mtx = confusion_matrix(y_test, y_pred)

# 予測結果と、正解（本当の答え）がどのくらい合っていたかを表す混合行列の標準化
x_sm=(sum(mtx[0]))
y_sm=(sum(mtx[1]))
x_arry=mtx[0]/x_sm*100
y_arry=mtx[1]/y_sm*100

n_mtx = x_arry,y_arry


#混同行列（％）のデータとカラムラベル挿入

index = [u'predict_'+smn1, u'predict_'+smn2]
columns = [u'real_'+smn1, u'real_'+smn2]

#混同行列（数）のデータとカラムラベル挿入
MX1 = pd.DataFrame(mtx, index=index, columns=columns)
MX2 = pd.DataFrame(n_mtx,index=index, columns=columns)


#モデルをもちいた予想の可視化
fig, ax = plt.subplots(figsize=(4, 3)) # 混合行列のカラムの大きさ設定
sns.heatmap(MX1, annot=True, fmt="d",center=250)
ax.set_ylim(len(mtx), 0)# 混合行列の軸の下限を設定し，値がみえるようにする（バグ）
ax.set_title('Confusion_Matrix')

#モデルをもちいた予想の可視化
fig, ax = plt.subplots(figsize=(4, 3)) # 混合行列のカラムの大きさ設定
sns.heatmap(MX2, annot=True, fmt="1.1f",center=250)# fmtでデータの表示桁数
ax.set_ylim(len(n_mtx), 0)# 混合行列の軸の下限を設定し，値がみえるようにする（バグ）
ax.set_title('normalized confusion matrix')

