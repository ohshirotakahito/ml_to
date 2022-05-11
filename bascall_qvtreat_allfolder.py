# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:07:30 2022

@author: ohshi
"""

import numpy as np
import csv
import pandas as pd
from tqdm import tqdm

def npypick(DataPath, SavePath):
    #dataの読み込み
    datum = np.load(DataPath, allow_pickle=True)

    #dataframe作成
    df = pd.DataFrame(data=datum)

    #ASCII_Qによる数値データ作成
    tx=df['Assigned Time']
    ty=df['S ASCII_Q']
    
    AD =[]
    for i in range(len(ty)):
        xx = ty[i]
        D=''
        for j in range(len(xx)):
            p = xx[j]
            d = ord(p)-33
            if j < len(xx)-1:
                D = D+str(d)+','
            else:
                D = D + str(d)
        AD.append(D)
    
    #dataframeの追加
    df['S ASCII_QV']=AD
    
    #dataの保存
    df.to_csv(SavePath, sep='\t')
    
    return AD

if __name__ =='__main__':
    #data 選択
    ServerPath = '//Rackstation/analysis/'
    in_folder = 'x_data'
    out_folder = 'ba_data'
    samples = ['PhiX 0001-0120_N']
    
    columns = ['File name','S#', 'Assinged Sequence', 'Assigned Time',
               'Signal Length [ms]', 'Signal Start [s]', 
               'Signal End [s]','Q value [x]', 'S ASCII_Q']
    
    for i in tqdm(samples):
        sample = i
        target = 'PhiX '+sample
        DataPath = in_folder +'/'+ sample+'.npy'
        SavePath = out_folder +'/'+ sample+'.tsv'
        
        #モジュール読み出し
        npypick(DataPath, SavePath)