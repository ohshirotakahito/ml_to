# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 13:40:06 2021

@author: ohshi
"""
import os
import glob
from BBrc import bnal_BC
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
    #対象となるファイルリストを作成
    files = glob.glob(Folderpath)
    
    
    CX=[]
    for file in files:
        BX=[]
        path_load = file
        #print(path_load)
        BX = bnal_BC(path_load)
        CX.extend(BX)
    
    #保存するパスを作成する
    save_path = ('data/'+SamplePath + '_' + TargetPath +'_'+'b.npy')
    #save_path1 = ('data/'+SamplePath + '_' + TargetPath +'_'+'b.csv')
    
    np.save(save_path, CX)
    #np.savetxt(save_path1, CX, delimiter=",", fmt="%.5f")
    
if __name__ =='__main__':
    ex = 'P53_F(OM)'
    Samples = ['OM1-01','OM1-02','OM1-03',
               'OM2-01','OM2-02','OM2-03',
               'OM3-01','OM3-02','OM3-03',
               'OM4-01','OM4-02','OM4-03',] 
    Targets = ['FTarget01','FTarget01_F_6','FTarget01_F_16']
    for sample in Samples:
        for target in Targets:
            Bnalasmb(ex,sample,target)
    print('end')