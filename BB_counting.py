# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 00:07:24 2021

@author: ohshi
"""
import numpy as np
import nptdms
from nptdms import TdmsFile
from nptdms import tdms
import glob
import os

def bnal_BC(path_load):

    #Tdmsファイル読み込み
    tdms_file = nptdms.TdmsFile(path_load)
    basename = os.path.splitext(os.path.basename(path_load))[0]
    
    #TDMSのツリー構造の取得
    #Tdmsファイルのツリー構造のGroups名取得
    tdms_groups = tdms_file.groups()
    ac = len(tdms_groups)
    
    AX=[]
    
    #tdms_groups[0]
    #print(tdms_groups[9])
    #group = tdms_groups[10]
    #print(group)
    
    if ac <=12:
        print(ac,': a_skipping',basename)
        #print(cc,': skipping',basename)
    else:
        #tdms_groups[0]
    #print(tdms_groups[9])
        group = tdms_groups[12]
        #print(group)
        
        #Tdmsファイルのツリー構造のChannel名取得
        channels0 =group.channels()
        bc = len(channels0)
        
        if bc ==0:
            print(bc,': b_skipping',basename)
            #print(cc,': skipping',basename)
        else:
            #tdms_groups[0]
            group = tdms_groups[12]
            print(bc,': pass',basename)
        
           #print('pass')
            #Tdmsファイルデータ読み込み（DMSの一行行列（時系列データ:raw data）の取得）
            
        
            #S_Table dataの再構成
            LAPB_data=tdms_file['LAP Table']['Assigned Base'].raw_data
            LAPX_data=tdms_file['LAP Table']['MS_IDX'].raw_data
            LAPR_data=tdms_file['LAP Table']['Relative Averaged Data'].raw_data
            LAPT_data=tdms_file['LAP Table']['Base Time length [ms]'].raw_data
        
        
            #S_Table dataの整数/少数/文字データ化
# =============================================================================
#             LAPB_data= [str(s) for s in LAPB_data]
#             LAPX_data= [int(s) for s in LAPX_data]
#             LAPR_data= [float(s) for s in LAPR_data]
#             LAPT_data= [float(s) for s in LAPT_data]
# =============================================================================
            
            
            
            T_LAPR_data=[]
            for r in LAPR_data:
                if r =='':
                    T_LAPR_data.append(float('0'))
                else:
                    T_LAPR_data.append(float(r))
            
            T_LAPB_data=[]
            for b in LAPB_data:
                if b =='':
                    T_LAPB_data.append('B')
                else:
                    T_LAPB_data.append(str(b))
        
            T_LAPX_data=[]
            for x in LAPX_data:
                if x =='':
                    T_LAPX_data.append(int('0'))
                else:
                    T_LAPX_data.append(int(x))
            
            T_LAPT_data=[]
            for t in LAPT_data:
                if t =='':
                    T_LAPT_data.append('0')
                else:
                    T_LAPT_data.append(float(t))
        
            #print(len(TT_data),len(BB_data))
            
            #S dataのビューア範囲絞り込み（テストのため）
            EX=[]
            for m in range(len(T_LAPR_data)):
                #EXの情報の記述
                e0=T_LAPX_data[m]
                e1=T_LAPB_data[m]
                e2=T_LAPR_data[m]
                e3=T_LAPT_data[m]
                
                EX=[e0,e1,e2,e3]
                
                AX.append(EX)
    
    AX= [str(s) for s in AX]
    #print(AX)
    return AX

if __name__ =='__main__':
    ex = 'KATO'
    sample = 'KTRNA01'
    target = 'hsa-miR-15a-5p'
    
    ServerPath = '//Rackstation/analysis/'
    DataPath = ex +'/' + sample+'/'
    SamplePath = sample+'_10k_Sample'
    TargetPath = 'BNAL@'+target
    FileForm ='/*.tdms'
    
    Folderpath = ServerPath + DataPath + SamplePath +'/T/' + TargetPath + FileForm
    files = glob.glob(Folderpath)
    #print(Folderpath)
    CX = []
    for file in files:
        BX=[]
        path_load = file
        #print(path_load)
        BX = bnal_BC(path_load)
        CX.append(BX)
    print(CX)
