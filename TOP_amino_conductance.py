# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:01:47 2021

@author: ohshi
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#サンプル名リスト
smns=['LAla','LAsp','LCys','LGlu','LLys','LPhe','LPro','LSer','LVal','Me1Lysine']

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

#データ読み込み
dnf = pd.DataFrame(columns=columns)
for smn in smns:
    sam = smn + '_10k_Sample_ANAL_r'
    datum = np.load('a_data/'+sam+'.npy',allow_pickle=True)
    df = pd.DataFrame(data=datum, columns=columns)
    dfx = df['signal_intensity'].astype('float')
    print(smn,':', len(df), dfx.quantile(0.80))