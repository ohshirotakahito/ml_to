# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 16:33:27 2021

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
    Samples = ['OM1-07','OM1-08','OM1-09',
               'OM2-07','OM2-08','OM2-09',
               'OM3-07','OM3-08','OM3-09',
               'OM4-07','OM4-08','OM4-09',] 
    Targets = ['FTarget01','FTarget01_F_6','FTarget01_F_16']
    for sample in Samples:
        for target in Targets:
            Bnalasmb(ex,sample,target)
    print('end')