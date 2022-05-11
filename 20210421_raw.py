#author: ohshirotakahito
#since: 202021/04/09

import numpy as np
import nptdms
import matplotlib.pyplot as plt
from nptdms import TdmsFile
from nptdms import tdms

#tdms_file = TdmsFile.read(r'"D:personal\python\MLUT_NTO\data\d.tdms")
path ="d.tdms"

#Tdmsファイル読み込み
tdms_file = nptdms.TdmsFile(path)

#TDMSのツリー構造の取得
#Tdmsファイルのツリー構造のGroups名取得
#tdms_groups = tdms_file.groups()

##tdms_groups[0]
#print(tdms_groups[0])

#Tdmsファイルのツリー構造のChannel名取得
#channels =tdms_groups[0].channels()
#print(channels)

#Tdmsファイルデータ読み込み（DMSの一行行列（時系列データ:raw data）の取得）
y_data=tdms_file['Example']['Ch1'].raw_data

#Tdmsファイルデータから時間行列の作製
#データ数の取得
y_n = len(y_data)
x_data =[]

#時間データの行列の作成と時間スケールの調整（ここでは10kHz）
#for i in range(y_n):
#    x = i*0.001
#    x_data.append(x)
    
#時間データと電流データによるプロット
#時間データと電流データのプロット範囲の設定
a=510000#開始データ点
b=10000#データポイント数
c=a+b

x_data = x_data[a:c]
y_data = y_data[a:c]


#時間データと電流データの移動平均からベースライン(b1)を作る
p1=2000#移動平均のデータポイント数
bl_data=[]

for j in range(len(y_data)):
    if j<p1:
        bl=np.average(y_data[0:j])
    else:
        bl=np.average(y_data[j-p1:j+p1])
    bl_data.append(bl)

y0_data = bl_data
y1_data = y_data
y2_data = y1_data-y0_data


#シグナル開始レベル
p2=500#標準偏差のデータポイント数
s_data=[]
s1_data=[]
for k in range(len(y2_data)):
    if k<p2:
        s1=np.std(y2_data[0:k+2])
    else:
        s1=np.std(y2_data[k-p2:k+p2])
    s_data.append(s1)
    s1_data.append(s1*6)
   

#シグナル開始レベル


#データ数の取得
y2_n = len(y2_data)
x_data =[]

#時間データの行列の作成と時間スケールの調整（ここでは10kHz）
for i in range(y2_n):
    x = (a+i)*0.001
    x_data.append(x)

#ファイルの時系列データの描画(raw dataとBaselineの同時描画)
plt.plot(x_data, y2_data)
plt.plot(x_data, s_data)
plt.plot(x_data, s1_data)
plt.show()

plt.show()


