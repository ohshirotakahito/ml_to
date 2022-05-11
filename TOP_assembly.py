# -*- coding: utf-8 -*-
"""
Created on Wed May 26 16:25:41 2021

@author: ohshi
"""
import os
import glob
from BB_r import bnal_RR
import nptdms
from nptdms import TdmsFile
from nptdms import tdms
import numpy as np

def Bnalasmb(ex,sample,target):
    ##対象となるフォルダを指定する．
    ServerPath = '//Rackstation/analysis/'
    DataPath = ex +'/'+sample+'/'
    SamplePath = sample+'_10k_Sample'
    TargetPath = 'BNAL@'+target
    FileForm ='/*.tdms'
    
    #対象となるフォルダパスを作成
    
    Folderpath = ServerPath + DataPath + SamplePath +'/T/' + TargetPath + FileForm
    #Folderpath = "./bnal_data/*"
    
    #print(Folderpath)
    
    
    files = glob.glob(Folderpath)
    
    
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
        channels0 =tdms_groups[8]
        channels1 =tdms_groups[8].channels()
        
        
        AX=[]
        
        ##エラーを起こすファイルの条件を設定し，それを除いたファイルでリストを作る
        cc = len(channels0)
        cb = len(channels1)
        if cb ==1:
            AX=[]
            print(cb,': skipping',basename)
        elif cb == 16952:
            AX=[]
            print(cb,': skipping',basename)
        elif cb == 0:
            AX=[]
        else:
            print(cb,': pass',':'+sample+'_'+target+':',basename)
            AX = bnal_RR(file)
            CX.extend(AX)
    
    #保存するパスを作成する
    save_path = ('data/'+SamplePath + '_' + TargetPath +'_'+'r.npy')
    
    np.save(save_path, CX)
    
if __name__ =='__main__':
    ex = 'KATO3'
    Samples = ['KTRNA01','KTRNA02','KTRNA03','KTRNA04','KTRNA05',
               'KTRNA06','KTRNA07','KTRNA08','KTRNA09','KTRNA10']
    Targets = ['hsa-miR-122-5p_M_11',
               'hsa-miR-155-5p_M_07',
               'hsa-miR-155-5p_M_12',
               'hsa-miR-205_-5p_M_02',
               'hsa-miR-205_-5p_M_03',
               'hsa-miR-205_-5p_M_06',
               'hsa-miR-205_-5p_M_10',
               'hsa-miR-205_-5p_M_11',
               'hsa-miR-205_-5p_M_13',
               'hsa-miR-205_-5p_M_14',
               'hsa-miR-205_-5p_M_20']
    for sample in Samples:
        for target in Targets:
            Bnalasmb(ex,sample,target)
    print('end')