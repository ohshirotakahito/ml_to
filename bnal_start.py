# -*- coding: utf-8 -*-
"""
Created on Tue May 25 12:05:24 2021

@author: ohshi
"""
import os
import glob
from BB_r import bnal_RR
import nptdms
from nptdms import TdmsFile
from nptdms import tdms
import numpy as np


##対象となるフォルダを指定する．

#試料名とターゲット配列を入力する以下のところを適宜修正）
sample = 'P53_RKOP'
target = 'P53up_F_7'

#Pathの作成要素を作成
ServerPath = '//Rackstation/analysis/'
DataPath = 'P53_RKO/'+sample+'/'
SamplePath = sample+'_10k_Sample'
TargetPath = 'BNAL@'+target
FileForm ='/*.tdms'

#対象となるフォルダを作成

Folderpath = ServerPath + DataPath + SamplePath +'/T/' + TargetPath + FileForm
#Folderpath = "./bnal_data/*"

#print(Folderpath)


files = glob.glob(Folderpath)

#ファイルリストの範囲を指定する（テスト用）
#files=files[0:10]

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
    #channels1 =tdms_groups[3].channels()
    
    
    AX=[]
    
    ##エラーを起こすファイルの条件を設定し，それを除いたファイルでリストを作る
    cc = len(channels0)
    if cc ==1:
        AX=[]
        #print(cc,': skipping',basename)
    elif cc == 16952:
        AX=[]
        #print(cc,': skipping',basename)
    elif cc == 0:
        AX=[]
    else:
        print(cc,': pass',basename)
        AX = bnal_RR(file)
        CX.extend(AX)

#保存するパスを作成する
save_path = (SamplePath + '_' + TargetPath +'_'+'r.npy')

np.save(save_path, CX)