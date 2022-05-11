# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:18:00 2021

@author: ohshi
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 13:34:00 2021

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

def mixedamin(file):
    file0 = file.split('_10k_Sample_ANAL_r')
    
    
    Lf = file0[0].split('L_')
    Df = file0[0].split('D_')
    
    Lfx = Lf[1].split('_')
    Dfx = Df[1].split('_')
    
    L_str = Lfx[0]
    D_str = Dfx[0]
    
    Lx=[]
    for i in range(len(L_str)):
        Lx.append(L_str[i])
    
    Dx=[]
    for i in range(len(D_str)):
        Dx.append(D_str[i])
        
    smns =[]
    for i in range(len(Dx)):
        ans = ab.a_abb2(Dx[i])
        smns.append(f'D{ans}')
        
    for i in range(len(Lx)):
        ans = ab.a_abb2(Lx[i])
        smns.append(f'L{ans}')
    
    Gx=[]
    if 'G' in file:
        Gx.append('G')
        g_ans = ab.a_abb2(Gx[0])
        smns.append(g_ans)
    
    print(smns)
    
    #サンプル名リスト
    #smns=['pSer','LSer','LAla']#'AAT','ACT','AGT','ATT','CA','CC','CG','CT','GA','GC','GG','GT','TA','TC','TG','TT'
    
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
    
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    
    
    #訓練用データをxgb.DMatrixで，XGBoost用のデータ型に変換
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    
    
    params = {'max_depth': 10, 
              'eta': 1,
              'objective': 'multi:softmax',
              'eval_metric': 'mlogloss', 
              'num_class': num_class}
    num_round = 200
    
    bst = xgb.train(params, dtrain, num_round)
    
    # save to JSON
    bst.save_model("model.json")
    
    y_pred = bst.predict(dtest)
    
    acc = f1_score(y_test, y_pred, average="micro")
    
    acc = '{:.3g}'.format(acc)
        
    mtx = confusion_matrix(y_test, y_pred)
    MX = pd.DataFrame(mtx, index=mtx_index, columns=mtx_columns)
    
    n_mtx=[mtx[i]/(sum(mtx[i]))*100 for i in range(len(mtx))]
    N_MX = pd.DataFrame(n_mtx, index=mtx_index, columns=mtx_columns)
    
    report = classification_report(y_test, y_pred)
    
    print('f-measure_value:', acc)
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
    sns.heatmap(N_MX, annot=True, fmt="1.1f",center=250)
    ax.set_ylim(len(n_mtx), 0)# 混合行列の軸の下限を設定し，値がみえるようにする（バグ）
    ax.set_title('Normalized_Confusion_Matrix')
    
    #混合サンプルのファイルのロード
    datum = np.load('a_data/'+file+'.npy',allow_pickle=True)
    
    dft = pd.DataFrame(data=datum, columns=columns)
    x_t = dft[data]
    x_t = preprocessing.scale(x_t)
    
    dtest_t = xgb.DMatrix(x_t)
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
    
    Dm = 0
    Lm = 0
    Gm = 0
    
    for i in spns:
        p = amin_c[i]
        if i==0:
            Dn = Dn + p
            Dm += 1
        elif i ==1:
            Ln = Ln + p
            Lm += 1 
        else:
            Gn = Gn + p
            Gm += 1
    
    Dp = '{:.3g}'.format(Dn/(Dn + Ln + Gn)*100)
    Lp = '{:.3g}'.format(Ln/(Dn + Ln + Gn)*100)
    Gp = '{:.3g}'.format(Gn/(Dn + Ln + Gn)*100)
    
    Dop = '{:.3g}'.format(Dm/(Dm + Lm + Gm)*100)
    Lop = '{:.3g}'.format(Lm/(Dm + Lm + Gm)*100)
    Gop = '{:.3g}'.format(Gm/(Dm + Lm + Gm)*100)

    
    
    print(f'D = {Dn}  D (%) = {Dp}  D_origin(%) = {Dop}')
    print(f'L = {Ln}  L (%) = {Lp}  L_origin(%) = {Lop}')
    print(f'G = {Gn}  G (%) = {Gp}  G_origin(%) = {Gop}')
    

if __name__ =='__main__':
    #混合アミノ酸サンプルフォルダ
    files = ['1L_R_D_NDQ_10k_Sample_ANAL_r',
             '1D_R_L_NDQ_10k_Sample_ANAL_r',
             '1G_L_W_D_TMN_10k_Sample_ANAL_r',
             '2D_Y_L_SHF_10k_Sample_ANAL_r',
             '2G_L_T_D_WMN_10k_Sample_ANAL_r',
             '2L_Y_D_SHF_10k_Sample_ANAL_r',
             '3G_L_M_D_WTN_10k_Sample_ANAL_r',
             '4G_L_N_D_WTM_10k_Sample_ANAL_r',
             '4D_K_L_ALE_10k_Sample_ANAL_r',
             '4L_K_D_ALE_10k_Sample_ANAL_r'
             ]
    file = '2G_L_T_D_WMN_10k_Sample_ANAL_r' 
    
    for f in files:
        mixedamin(f)
