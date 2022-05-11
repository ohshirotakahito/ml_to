"""
Created on Fri Apr  1 16:09:51 2022

@author: ohshi
"""

import numpy as np
import pandas as pd
import glob
import os
import re

import nptdms
import matplotlib.pyplot as plt
import math
from nptdms import TdmsFile
from nptdms import tdms

from tqdm import tqdm

def bC_sq(files):
    #Tdmsファイル中の選択columnｓの指定
    columns = ['File name','S#', 'Assinged Sequence', 'Assigned Time',
               'Signal Length [ms]', 'Signal Start [s]', 
               'Signal End [s]','Q value [x]']
    
    #Tdmsファイル読み込み
    PX=[]
    for file in tqdm(files):
        path_load = file
        
        f= os.path.basename(path_load)
        ff=re.split(sample,f)
        f_name = sample + ff[1].split('.')[0]
    
        #Tdmsファイル読み込み
        tdms_file = nptdms.TdmsFile(path_load)
        
        try:
            #BS TableのDataframe化
            dfac = tdms_file['BS Table'].as_dataframe()
            
            #データ選択('Assinged Seqeuence'のn連続以上)の準備
            as_df = dfac['Assinged Sequence']
            as_list = as_df.values.tolist()
            n_n = 5
            
            #データ選択の選択行リストの作成と選択
            u_list=[]
            for i, name in enumerate(as_list):
                ni = len(name)
                if ni>n_n:
                    u_list.append(i)
            
            #データ選択の選択行によるDataframeの作成
            dfac_r = dfac.iloc[u_list]
            dfac_r = dfac_r[columns]
            xs = dfac_r.to_records()
            pxi ='yes_signal: '
            
            ppx = []
            for x in xs:
                #print(x)
                ppx.append(x)
            PX=PX+ppx

        except KeyError:
            pxi ='no_signal: '
            
        print(pxi + f_name)
    return PX


if __name__ =='__main__':
    #data 選択
    ServerPath = '//Rackstation/analysis/'
    ex = 'Phi174'
    samples = ['0001-0120']
    for i in samples:
        sample = i
        target = 'PhiX '+sample
        DataPath = ex +'/'+ sample+'/'
        SamplePath = sample +'_10k_Sample'
        TargetPath = 'BNAL@'+target
        FileForm ='/*.tdms'
        
        #file list 作成
        Folderpath = ServerPath + DataPath + SamplePath +'/T/' + TargetPath + FileForm
        files = glob.glob(Folderpath)
        
        #モジュール読み出し
        PX = bC_sq(files)
        
        #モジュール読み出し
        save_path = 'x_data/'+target+'_N.npy'
        np.save(save_path, PX)