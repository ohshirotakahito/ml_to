# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 00:05:39 2021

@author: ohshi
"""
import os
import glob
import nptdms
from nptdms import TdmsFile
from nptdms import tdms
import numpy as np
import BBrc
import csv

def Bnalrcount(Folderpath):
    files = glob.glob(Folderpath)
    print(len(files))
    #ファイルリストの範囲を指定する（テスト用）
    #files=files[0:10]
    
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
            AX = BBrc.bnal_CC(path_load)
            print(cc,': pass',':'+sample+':',basename)
            CX.extend(AX)
        
    return(CX)
                
if __name__ =='__main__':
    sample = 'DRD22_97'
    target = 'DRD2_97'
    
    ServerPath = '//Rackstation/analysis/'
    DataPath = 'LG_EX/'+sample+'/'
    SamplePath = sample+'_10k_Sample'
    TargetPath = 'BNAL@'+target
    FileForm ='/*.tdms'
    
    Folderpath = ServerPath + DataPath + SamplePath +'/T/' + TargetPath + FileForm
    
    CX=Bnalrcount(Folderpath)
    #print(CX)
    save_path = ('a_data/'+SamplePath + '_' + TargetPath +'_'+'r.csv')
    with open(save_path, 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL,delimiter=';')
        writer.writerows(CX)