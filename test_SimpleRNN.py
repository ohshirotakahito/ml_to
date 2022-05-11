# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 13:21:32 2021

@author: ohshi
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3*np.pi,3*np.pi,1000)
y = np.sin(x)
#plt.plot(x, y) # 確認。別にプロットしなくていい

input_len = 5 # 入力の長さ
X, Y = [], []
for i, _ in enumerate(x):
    if (i+input_len+1 >= len(x)):
        break
    X.append(y[i:i+input_len])
    Y.append(y[i+input_len+1])

split_index = int(len(X)*0.8)

train_x = X[:split_index]
train_y = Y[:split_index]
test_x = X[split_index:]
test_y = Y[split_index:]

train_x = np.array(train_x).reshape(len(train_x),-1,1)
test_x = np.array(test_x).reshape(len(test_x), -1, 1)
train_y = np.array(train_y).reshape(-1,1)
test_y = np.array(test_y).reshape(-1,1)

from keras.layers import Dense, Activation, SimpleRNN
from keras.models import Sequential

input_shape = (len(train_x[0]), 1)
model = Sequential()
model.add(SimpleRNN(100, return_sequences=False, input_shape=input_shape))
model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss="mse", optimizer="adam")

#plt.plot(np.arange(len(test_y)), test_y, label="sin")
predict_y = model.predict(test_x)

hist = model.fit(train_x, train_y, batch_size=32, epochs=100, verbose=0)
history = hist.history
plt.plot(hist.epoch, history["loss"], label="loss")
plt.show()

# plt.plot(np.arange(len(test_y)), predict_y, label="predict(before)")
# plt.legend()
# plt.show()

score = model.evaluate(test_x, test_y)
print(score)

plt.plot(np.arange(len(test_y)), test_y, label="sin")
predict_y = model.predict(test_x)
plt.plot(np.arange(len(test_y)), predict_y, label="predict")
plt.legend()
plt.show()