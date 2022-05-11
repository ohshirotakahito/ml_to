# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 13:20:25 2021

@author: ohshi
"""
import numpy as np
import nptdms
import matplotlib.pyplot as plt
import math
from nptdms import TdmsFile
from nptdms import tdms
from tqdm import tqdm
import fea_para
import os

#tdms_file = TdmsFile.read(r'"D:personal\python\MLUT_NTO\data\d.tdms")

def apick(path_load):
    try:
        #Tdmsファイル読み込み
        tdms_file = nptdms.TdmsFile(path_load)
        
        #TDMSのツリー構造の取得
        #Tdmsファイルのツリー構造のGroups名取得
        tdms_groups = tdms_file.groups()
        
        tdms_groups[0]
        #print(tdms_groups[0])
        
        #Tdmsファイルのツリー構造のChannel名取得
        channels0 =tdms_groups[0].channels()
        #print(channels0)
        channels1 =tdms_groups[3].channels()
        #print(channels1)
        
        #Tdmsファイルデータ読み込み（DMSの一行行列（時系列データ:raw data）の取得）
        Y_data=tdms_file['Data']['Ch1'].raw_data
        
        #Tdmsファイルデータ読み込み（DMSの一行行列（時系列データ:raw data）の取得）
        Data_name = tdms_file['AR Table']['Filename'].raw_data
        #Device_name = tdms_file['AR Table']['EP2 Voltage (mV)'].raw_data
        #Ex_ID = tdms_file['AR Table']['EP2 Voltage (mV)'].raw_data
        Distance = tdms_file['AR Table']['Distance (nm)'].raw_data
        Ex_ID = tdms_file['AR Table']['Ex ID'].raw_data
        
        target0 = ' D_'
        F_name = Data_name[0]
        idx0 = F_name.find(target0)
        
        Sample_name = F_name[:idx0+len(target0)-3]
        e0 = Sample_name
           
        Ex_ID = Ex_ID[0]
        e1 = Ex_ID
        
        Distance = Distance[0]
        e2 = Distance
        
        basename = os.path.basename(path_load)
        e3= basename.split('_')[4]##
    # =============================================================================
    #     target1 = 'DV_'
    #     Device_name = Device_name[0]
    #     idx1 = Device_name.find(target1)
    #     Gap_ID = Device_name[idx1+len(target1):]
    #     
    #     target2 = '-G'
    #     idx2 = Gap_ID.find(target2)
    #     Device_ID = Gap_ID[:idx2+len(target2)-2]
    #     
    #     target3 = ' A'
    #     idx3 = Ex_ID.find(target3)
    #     Date_ID = Ex_ID[:idx3+len(target3)-2]
    #     PDate = Ex_ID[:idx3+len(target3)-1]
    #     
    #     target4 = 'Pex'
    #     idx4 = PDate.find(target4)
    #     Machine_ID = PDate[:idx4+len(target4)-3]
    # =============================================================================
        
        #S_Table dataの再構成
        #SM_data=tdms_file['S Table']['S # [n]'].raw_data
        SP_data=tdms_file['S Table']['S Peak Position [s]'].raw_data
        SI_data=tdms_file['S Table']['Signal [pA]'].raw_data
        SB_data=tdms_file['S Table']['Region BL [pA]'].raw_data
        SD_data=tdms_file['S Table']['Region STD [pA]'].raw_data
        SS_data=tdms_file['S Table']['Signal S [s]'].raw_data
        SE_data=tdms_file['S Table']['Signal E (s)'].raw_data
        ST_data=tdms_file['S Table']['S TL [ms]'].raw_data
        SL_data=tdms_file['S Table']['S DL [s]'].raw_data
        
        #S_Table dataの少数データ化
        #SM_data= [int(s) for s in SM_data]
        SP_data= [float(s) for s in SP_data]
        SI_data= [float(s) for s in SI_data]
        SB_data= [float(s) for s in SB_data]
        SD_data= [float(s) for s in SB_data]
        SS_data= [float(s) for s in SS_data]
        SE_data= [float(s) for s in SE_data]
        ST_data= [float(s) for s in ST_data]
        SL_data= [float(s) for s in SL_data]
        #print(len(SI_data))
           
        
        AX=[]
        count =0
    
        for m in tqdm(range(len(SP_data))):
            XX=[]
            ssp=[]
            #sm = SM_data[m]
            sp = SP_data[m]*10000
            ss = SS_data[m]*10000
            se = SE_data[m]*10000
            st = ST_data[m]*10
            si = SI_data[m]
            sb = SB_data[m]
            sd = SD_data[m]
            sl = SL_data[m]*10000
            #print(st)
                
            sp= int(sp)
            ss= int(ss)
            se= int(se)
            st= int(st)
            
            sb=round(sb,6)
            sd=round(sd,6)
            sl=round(sl,6)
            
            if st>10:#default =10
                ssp = fea_para.fp(Y_data, st, sb, ss, se, si)
                #print(ssp)
                count+=1
                XX= [e0, e2, e3, sp, si, st, ss, se, sb]+ssp
                AX.append(XX)
        return(AX)
    except ValueError as e:
        print(e)
        AX=[]
        return(AX)

def a_countpick(path_load):
    try:
        #Tdmsファイル読み込み
        tdms_file = nptdms.TdmsFile(path_load)
        
        n = tdms_file['AR Table']['Filename'].raw_data
        B_count = tdms_file['AR Table']['Burst#Sum'].raw_data
        S_count = tdms_file['AR Table']['Signal#Sum'].raw_data
        
        n = n[0][:n[0].find(' D_')]
        b=int(B_count[0])
        s=int(S_count[0])
        
        AZ=[]
        AZ=[n,b,s]
        return(AZ)
    
    except ValueError as e:
        print(e)
        AZ=[]
        return(AZ)

if __name__ =='__main__':
    ServerPath = '//Rackstation/analysis/'
    ex = 'KATO'
    sample = 'KTRNA01'
    DataPath = ex +'/'+sample+'/'
    SamplePath = sample+'_10k_Sample'
    TargetPath = 'ANAL'
    FileForm ='/A@@2021_1022_1636_11_KTRNA01_10k_Sample#001.tdms'
    
    #対象となるフォルダパスを作成
    
    path_load = ServerPath + DataPath + SamplePath +'/T/' + TargetPath + FileForm
    CC = apick(path_load)
    #AC = a_countpick(path_load)
    print(CC)