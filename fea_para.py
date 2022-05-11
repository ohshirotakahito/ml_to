# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:06:32 2021

@author: ohshi
"""
import numpy as np
import nptdms
import matplotlib.pyplot as plt
import math
from nptdms import TdmsFile
from nptdms import tdms

#tdms_file = TdmsFile.read(r'"D:personal\python\MLUT_NTO\data\d.tdms")
#path_load ="a.tdms"

#Tdmsファイル読み込み

def fp(Y_data, st, sb, ss, se, si):
    mr = int(st/2)
    y0_data = Y_data[ss-1-mr:se+1+mr]
    
     #シグナル分割数
    sn=13
    
    #シグナルベクトル作成：シグナル分割案１(np.array_split)
    #ssl=np.array_split(y0_data,sn)
    #for u in range(sn):
       #print('s_region',u,':',ssl[u])
    
    #シグナルベクトル作成：シグナル分割2(シグナル2領域)
    sru=len(y0_data)/sn
    y=[]
    PX=[]
    for u in range(sn-1):
        x=sru*u
        x=round(x,5)
        mc=math.ceil(sru*u)
        mf=math.floor(sru*u)
        x1=mf
        if mc==len(y0_data):
            mc==len(y0_data)-1
            
        x2=mf+1
        mz=mc-mf
        #rint(x,mc,mf,mz)
        if mz<1:
            y1=y0_data[mf]
            y2=y0_data[mc]
            y=y1
        else:
            y1=round(y0_data[mf],5)
            y2=round(y0_data[mc-1],5)
            y=(y2-y1)/(x2-x1)*(x-x1)+y1
                    
        y=round(y*1000-sb,5)
        y=round(y/si,5)
        y1=round(y1*1000-sb,5)
        y2=round(y2*1000-sb,5)
        x=round(x,5)
        PX.append(y)
    return(PX)
    
if __name__ =='__main__':
    #Tdmsファイル読み込み
    path_load ="a.tdms"
    tdms_file = nptdms.TdmsFile(path_load)
    
    Y_data = tdms_file['Data']['Ch1'].raw_data
    
    st = 46
    sb = 10.6883
    ss = 4796896
    se = 4796942
    si = 19.4938

    CC = fp(Y_data, st, sb, ss, se, si)
    print(CC)
