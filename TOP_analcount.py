# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 10:33:33 2021

@author: ohshi
"""
import os
import glob
import nptdms
from nptdms import TdmsFile
from nptdms import tdms
import numpy as np
import sig_anal
from tqdm import tqdm

def fe_xx(ex,sample):
    ##対象となるフォルダを指定する．
    ServerPath = '//Rackstation/analysis/'
    DataPath = ex +'/'+sample+'/'
    SamplePath = sample+'_10k_Sample'
    TargetPath = 'ANAL'
    FileForm ='/*.tdms'
    
    #対象となるフォルダパスを作成
    
    Folderpath = ServerPath + DataPath + SamplePath +'/T/' + TargetPath + FileForm
    #Folderpath = "./bnal_data/*"
        
    files = glob.glob(Folderpath)
    
    CZ=[]
    bx0 = 0
    bx1 = 0
    for file in tqdm(files):
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
        channels0 =tdms_groups[3]
        #channels1 =tdms_groups[3].channels()
        cc = len(channels0)
        f_name = os.path.splitext(os.path.basename(path_load))[0]
        #print(f_name)
        if cc ==0:
                bz = 0
                sz = 0
                
                #print(cc,': skipping',basename)
        else:
            AZ = sig_anal.a_countpick(file)
            bz = AZ[1]
            sz = AZ[2]
        
        bx0 = bx0 + bz
        bx1 = bx1 + sz
    return bx0, bx1
# =============================================================================
#     #保存するパスを作成する
#     save_path = ('a_data/'+SamplePath + '_' + TargetPath +'_'+'r.npy')
#     np.save(save_path, CZ)
#         
# =============================================================================
            
if __name__ =='__main__':
    ex = 'Gynecologic Cancer'
    Samples =['BB01']#'AAT','ACT','AGT','ATT','CA','CC','CG','CT','GA','GC','GG','GT','TA','TC','TG','TT'
    for sample in Samples:
        XX=fe_xx(ex, sample)
        print(sample, XX)
    print('end')