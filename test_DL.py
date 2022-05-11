# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 11:26:11 2021

@author: ohshi
"""
#参考ページ
#http://aidiary.hatenablog.com/category/Keras?page=1478696865

#モジュール読み出し
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from sklearn import preprocessing
from keras.layers import Dense, Dropout

# sklearnのwineデータの読み込み
wine = datasets.load_wine()

X=wine.data
Y=wine.target

# データの標準化
X = preprocessing.scale(X)

# ラベルをone-hot-encoding形式に変換
X=wine.data
Y=wine.target
# 0 => [1, 0, 0]
# 1 => [0, 1, 0]
# 2 => [0, 0, 1]
    #risのラベルは文字列だがsklearnのデータセットでは0, 1, 2のように数値ラベルに変換されている。
    #これをニューラルネットで扱いやすいone-hotエンコーディング型式に変換する。
    #one-hotエンコーディングは、特定のユニットのみ1でそれ以外は0のようなフォーマットのこと。
    #この変換は、keras.utils.np_utils の to_categorical() に実装されている

# 訓練データとテストデータに分割
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.8)
         #shapeは，numpyのクラスndarryの変数　ndarray.shape

# 訓練データとテストデータのshapeをチェック
print(train_X.shape, test_X.shape, train_Y.shape, test_Y.shape)
         #ここで得た値をbuild_multilayer_perceptron()の値に導入

# モデル構築（訓練プロセスの定義）
model = Sequential()
model.add(Dense(16,input_shape=(13, )))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Dropout(0.2))
model.add(Activation('softmax'))
##dropoutは，汎化性能を上げ、過学習を避ける
#入力層が13ユニット、隠れ層が16ユニット、出力層が3ユニットの多層パーセプトロンを構築した

# モデル構築（訓練プロセスの定義）
model.compile(optimizer='adam',##学習の最適化法を決定(勾配法)
                  loss='categorical_crossentropy',#損失関数(誤差関数)を決定
                  metrics=['accuracy'])

# モデル訓練
history=model.fit(train_X, train_Y, epochs=10, batch_size=1, verbose=1)

# モデル評価
loss, accuracy = model.evaluate(test_X, test_Y, verbose=0)
print("Accuracy = {:.2f}".format(accuracy))
    
plt.plot(history.history['loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

#モデルをもちいた予想
from sklearn.metrics import confusion_matrix

predict_classes = model.predict_classes(test_X[1:100,], batch_size=32)
true_classes = np.argmax(test_Y[1:100],1)
w=confusion_matrix(true_classes, predict_classes)

#モデルをもちいた予想の可視化
fig, ax = plt.subplots(figsize=(4, 3)) # 混合行列のカラムの大きさ設定
sns.heatmap(w, annot=True, fmt="d")
ax.set_ylim(len(w), 0)# 混合行列の軸の下限を設定し，値がみえるようにする（バグ）
ax.set_title('result_')