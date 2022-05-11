# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:06:39 2021

@author: ohshi
"""
import os


def fo_xx(ex,sample):
    ##対象となるフォルダを指定する．
    ServerPath = '//Rackstation/analysis/'
    DataPath = ex +'/'+sample
    SamplePath = sample+'_10k_Sample'
    BlankPath = sample+'_10k_Blank'
    SamplePath1 = sample+'_100k_Sample'
    BlankPath2 = sample+'_100k_Blank'
    RawPath = 'raw data'
    
    #対象となるフォルダパスを作成
    Folderpath1 = ServerPath + DataPath
    Folderpath2 = ServerPath + DataPath +'/'+ SamplePath
    Folderpath3 = ServerPath + DataPath +'/'+ SamplePath1
    Folderpath4 = ServerPath + DataPath +'/'+ BlankPath
    Folderpath5 = ServerPath + DataPath +'/'+ BlankPath2
    Folderpath6 = ServerPath + DataPath +'/'+ RawPath
    Folderpath7 = ServerPath + DataPath +'/'+ SamplePath+'/T'
    Folderpath8 = ServerPath + DataPath +'/'+ BlankPath+'/T'
    Folderpath9 = ServerPath + DataPath +'/'+ SamplePath+'/stocked'
    Folderpath10 = ServerPath + DataPath +'/'+ BlankPath+'/stocked'
    Folderpath11 = ServerPath + DataPath +'/'+ SamplePath+'/T/ANAL'
    Folderpath12 = ServerPath + DataPath +'/'+ SamplePath+'/T/stocked'
    Folderpath13 = ServerPath + DataPath +'/'+ BlankPath+'/T/ANAL'
    Folderpath14 = ServerPath + DataPath +'/'+ BlankPath+'/T/stocked'
    
    #対象となるフォルダパスをリスト作成
    Folders = [Folderpath1,Folderpath2,Folderpath3,Folderpath4,Folderpath5,
               Folderpath6,Folderpath7,Folderpath8,Folderpath9,Folderpath10,
               Folderpath11,Folderpath12,Folderpath13,Folderpath14]
    
    for folder in Folders:
        os.mkdir(folder)
        print(folder)
    
            
if __name__ =='__main__':
    ex = 'Natsume'
    Samples =['IDH1MTBT142']
    for sample in Samples:
        XX=fo_xx(ex, sample)
    print('end')