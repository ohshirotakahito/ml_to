# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:19:13 2021

@author: ohshi
"""
import os
import glob
import nptdms
from nptdms import TdmsFile
from nptdms import tdms
import numpy as np
import AA_r
from tqdm import tqdm

def Analcount(sample):
    ##対象となるフォルダを指定する．
    
    #Pathの作成要素を作成
    ServerPath = '//Rackstation/analysis/'
    DataPath = 'Gynecologic Cancer'+sample+'/'
    SamplePath0 = sample+'_10k_Blank'
    SamplePath1 = sample+'_10k_Sample'
    TargetPath = 'ANAL'
    FileForm ='/*.tdms'
    
    #対象となるフォルダを作成
    
    Folderpath0 = ServerPath + DataPath + SamplePath0 +'/T/' + TargetPath + FileForm
    Folderpath1 = ServerPath + DataPath + SamplePath1 +'/T/' + TargetPath + FileForm
    #Folderpath = "./bnal_data/*"
    Folderpaths = (Folderpath0, Folderpath1)
    
    for Folderpath in Folderpaths:
        #print(Folderpath)
        
        files = glob.glob(Folderpath)
        print(len(files))
        #ファイルリストの範囲を指定する（テスト用）
        #files=files[0:10]
        #for m in tqdm(range(len(LN_data))):
        CX=[]
        for file in files:
            path_load = file
            basename = os.path.splitext(os.path.basename(path_load))[0]
            tdms_file = nptdms.TdmsFile(path_load)
        
            ##エラーを起こすファイルの条件を取得しておく
            #TDMSのツリー構造の取得
            #Tdmsファイルのツリー構造のGroups名取得
            tdms_groups = tdms_file.groups()
            #print(tdms_groups)
            
            #tdms_groups[0]
            #print(tdms_groups[0])
            
            #Tdmsファイルのツリー構造のChannel名取得
            channels0 =tdms_groups[3]
            #channels1 =tdms_groups[3].channels()
            
            
            AX=[]
            
            ##エラーを起こすファイルの条件を設定し，それを除いたファイルでリストを作る
            cc = len(channels0)
            #print(cc)
            if cc ==0:
                AX=[]
                #print(cc,': skipping',basename)
            elif cc == 167:
                AX=[]
                #print(cc,': skipping',basename)
            else:
                AX = AA_r.AAcount(path_load)
                print(cc,': pass',':'+sample+':',basename)
                CX.extend(AX)
    
    return(CX)
                
if __name__ =='__main__':
    sample = 'BB01' 
    CX = Analcount(sample)
    print(CX)