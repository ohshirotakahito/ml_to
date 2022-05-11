# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:29:44 2021

@author: ohshi
"""
import glob


def Analextract(ex, sample, target):
    ##対象となるフォルダを指定する．
    
    #Pathの作成要素を作成
    ServerPath = '//Rackstation/analysis/'
    DataPath = ex +'/'+sample+'/'
    SamplePath = sample+'_10k_Sample'
    TargetPath = 'ANAL'
    FileForm ='/*.tdms'
    
    #対象となるフォルダを作成
    
    Folderpath = ServerPath + DataPath + SamplePath +'/T/' + TargetPath + FileForm
    #Folderpath = "./bnal_data/*"
    
    files = glob.glob(Folderpath)
    return(files)

if __name__ =='__main__':
    ex = '200c(OM)'
    sample = '200c_native_C9_000' 
    target = 'FTarget01'
    files = Analextract(ex, sample, target)
    print(files)
