# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:12:12 2021

@author: ohshi
"""
import numpy as np
import nptdms
import matplotlib.pyplot as plt
import math
from nptdms import TdmsFile
from nptdms import tdms

#tdms_file = TdmsFile.read(r'"D:personal\python\MLUT_NTO\data\d.tdms")

def acount(path_load):
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
    #Y_data=tdms_file['Data']['Ch1'].raw_data
    
    #Tdmsファイルデータ読み込み（DMSの一行行列（時系列データ:raw data）の取得）
    Data_name = tdms_file['AR Table']['Filename'].raw_data
    Device_name = tdms_file['AR Table']['EP2 Voltage (mV)'].raw_data
    Ex_ID = tdms_file['AR Table']['EP2 Voltage (mV)'].raw_data
    Distance = tdms_file['AR Table']['Distance (nm)'].raw_data
    Ex_ID = tdms_file['AR Table']['Ex ID'].raw_data
    
    Distance = Distance[0]
    e2 = Distance
    
    Ex_ID = Ex_ID[0]
    e1 = Ex_ID
    
    target0 = ' D_'
    F_name = Data_name[0]
    idx0 = F_name.find(target0)
    Sample_name = F_name[:idx0+len(target0)-3]
    e0 = Sample_name
    
    target1 = 'DV_'
    Device_name = Device_name[0]
    idx1 = Device_name.find(target1)
    Gap_ID = Device_name[idx1+len(target1):]
    
    target2 = '-G'
    idx2 = Gap_ID.find(target2)
    Device_ID = Gap_ID[:idx2+len(target2)-2]
    
    target3 = ' A'
    idx3 = Ex_ID.find(target3)
    Date_ID = Ex_ID[:idx3+len(target3)-2]
    PDate = Ex_ID[:idx3+len(target3)-1]
    
    target4 = 'Pex'
    idx4 = PDate.find(target4)
    Machine_ID = PDate[:idx4+len(target4)-3]
    
    #S_Table dataの再構成
    SM_data=tdms_file['S Table']['S # [n]'].raw_data
    SP_data=tdms_file['S Table']['S Peak Position [s]'].raw_data
    SI_data=tdms_file['S Table']['Signal [pA]'].raw_data
    SB_data=tdms_file['S Table']['Region BL [pA]'].raw_data
    SD_data=tdms_file['S Table']['Region STD [pA]'].raw_data
    SS_data=tdms_file['S Table']['Signal S [s]'].raw_data
    SE_data=tdms_file['S Table']['Signal E (s)'].raw_data
    ST_data=tdms_file['S Table']['S TL [ms]'].raw_data
    SL_data=tdms_file['S Table']['S DL [s]'].raw_data
    
    #S_Table dataの少数データ化
    SM_data= [int(s) for s in SM_data]
    SP_data= [float(s) for s in SP_data]
    SI_data= [float(s) for s in SI_data]
    SB_data= [float(s) for s in SB_data]
    SD_data= [float(s) for s in SB_data]
    SS_data= [float(s) for s in SS_data]
    SE_data= [float(s) for s in SE_data]
    ST_data= [float(s) for s in ST_data]
    SL_data= [float(s) for s in SL_data]
       
    
    AX=[]
    count = 0
    
    for m in range(len(SM_data)):
        sm = SM_data[m]
        st = ST_data[m]*10
        si = SI_data[m]
        sd = SD_data[m]
        st= int(st)
        sd=round(sd,6)
    
        if st>=10:
            count = count+1
    
    e3 = len(SM_data)
    e4 = count    
    AX += [e0, e1, e2, e3, e4]
    
    AX= [str(s) for s in AX]
    
    return(AX)

if __name__ =='__main__':
    path_load ="a.tdms"
    CC = acount(path_load)
    print(CC)
