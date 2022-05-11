# -*- coding: utf-8 -*-
"""
Created on Tue May 25 12:05:24 2021

@author: ohshi


"""
import numpy as np
import nptdms
from nptdms import TdmsFile
from nptdms import tdms
import glob
from tqdm import tqdm

def bnal_RR(path_load):

    #Tdmsファイル読み込み
    tdms_file = nptdms.TdmsFile(path_load)
    
    #TDMSのツリー構造の取得
    #Tdmsファイルのツリー構造のGroups名取得
    #tdms_groups = tdms_file.groups()
    #print(tdms_groups)
    
    #tdms_groups[0]
    #print(tdms_groups[0])
    
    #Tdmsファイルのツリー構造のChannel名取得
    #channels0 =tdms_groups[8].channels()
    #print(channels0)
    #channels1 =tdms_groups[3].channels()
    
    AX=[]
    

   #print('pass')
    #Tdmsファイルデータ読み込み（DMSの一行行列（時系列データ:raw data）の取得）
    Data_name = tdms_file['AR Table']['Filename'].raw_data
    Device_name = tdms_file['AR Table']['EP2 Voltage (mV)'].raw_data
    Ex_ID = tdms_file['AR Table']['EP2 Voltage (mV)'].raw_data
    Distance = tdms_file['AR Table']['Distance (nm)'].raw_data
    Ex_ID = tdms_file['AR Table']['Ex ID'].raw_data
    
    Distance = Distance[0]
    Ex_ID = Ex_ID[0]
    
    target0 = ' D_'
    F_name = Data_name[0]
    idx0 = F_name.find(target0)
    Sample_name = F_name[:idx0+len(target0)-3]
    
    target1 = 'DV_'
    Device_name = Device_name[0]
    idx1 = Device_name.find(target1)
    Gap_ID = Device_name[idx1+len(target1):]
    
    target2 = '-G'
    idx2 = Gap_ID.find(target2)
    Device_ID = Gap_ID[:idx2+len(target2)-2]
    
    target3 = ' A'
    idx3 = Ex_ID.find(target3)
    PDate = Ex_ID[:idx3+len(target3)-1]
    
    target4 = 'Pex'
    idx4 = PDate.find(target4)
    Machine_ID = PDate[:idx4+len(target4)-3]

    #S_Table dataの再構成
    LN_data=tdms_file['LSP Table']['L#'].raw_data
    LQ_data=tdms_file['LSP Table']['Assinged Sequence'].raw_data
    LSN_data=tdms_file['LSP Table']['MSeq_SNo'].raw_data
    LEN_data=tdms_file['LSP Table']['MSeq_ENo'].raw_data
    LMR_data=tdms_file['LSP Table']['M_RRead_Seq'].raw_data
    LMX_data=tdms_file['LSP Table']['M_MxRight_No'].raw_data
    #ST_data=tdms_file['LSP Table']['S TL [ms]'].raw_data
    #SL_data=tdms_file['LSP Table']['S DL [s]'].raw_data
    
    #S_Table dataの整数/少数/文字データ化
    LN_data= [int(s) for s in LN_data]
    LQ_data= [str(s) for s in LQ_data]
    LSN_data= [int(s) for s in LSN_data]
    LEN_data= [int(s) for s in LEN_data]
    LMR_data= [str(s) for s in LMR_data]
    LMX_data= [int(s) for s in LMX_data]
    #ST_data= [float(s) for s in ST_data]
    #SL_data= [float(s) for s in SL_data]
    
    #print(LQ_data)
    
    AX=[]
    
    #S dataのビューア範囲絞り込み（テストのため）
    for m in tqdm(range(len(LN_data))):
        EX=[]
        #EXの情報の記述
        #print('+++++++++++++++++name++++++++++++++++++++++++')
        #print('E1:Ex_ID:',Ex_ID)
        e1=Ex_ID
        #EX.append(e1)
        #print('E2:Gap_ID:',Gap_ID)
        e2=Gap_ID
        #EX.append(e2)
        #print('E3:Device_ID:',Device_ID)
        e3=Device_ID
        #EX.append(e3)
        #print('E4:PDate:',PDate)
        e4=PDate
        #EX.append(e4)
        #print('E5:Machine_ID:',Machine_ID)
        e5=Machine_ID
        #EX.append(e5)
        #print('E6:Distance(nm):',Distance)
        e6=Distance
        #EX.append(e6)
        #print('E7:Sample_name:',Sample_name)
        e7=Sample_name
        EX.append(e7)
        #print('E8:L#:',LN_data[m])
        e8=LN_data[m]
        EX.append(e8)
        #print('E9:Assinged Sequence:',LQ_data[m])
        e9=LQ_data[m]
        EX.append(e9)
        #print('E10:Right_Read_Start_Position:',LSN_data[m])
        e10=LSN_data[m]
        EX.append(e10)
        #print('E11:Right_Read_End_Position:',LEN_data[m])
        e11=LEN_data[m]
        EX.append(e11)
        #print('E12:Right_Read',LMR_data[m])
        e12=LMR_data[m]
        EX.append(e12)
        #print('E13:Right_Read_Max:',LMX_data[m])
        e13=LMX_data[m]
        EX.append(e13)
        
        AX.append(EX)
    
    AX= [str(s) for s in AX]
    #print(AX)
    return AX

if __name__ =='__main__':
    ex = 'Methlylation'
    sample = 'M4MC'
    target = 'M4MC'
    n = 1
    
    ServerPath = '//Rackstation/analysis/'
    DataPath = ex +'/'+ sample+'/'
    SamplePath = sample+'_10k_Sample'
    TargetPath = 'BNAL@'+target
    FileForm ='/*.tdms'
    
    Folderpath = ServerPath + DataPath + SamplePath +'/T/' + TargetPath + FileForm
    files = glob.glob(Folderpath)
    print(Folderpath)
    #print (files[n])
    
    path_load = files[n]
    
    AX = bnal_RR(path_load)
    print(AX)
