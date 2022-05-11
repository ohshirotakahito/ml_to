# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:07:30 2021

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

def fe_xx(ExPath,sample):
    ##対象となるフォルダを指定する．
    SamplePath = sample+'_10k_Sample'
    TargetPath = 'ANAL'
    FileForm ='/*.tdms'
    
    #対象となるフォルダパスを作成
    
    Folderpath = ExPath + sample + '/' +SamplePath +'/T/' + TargetPath + FileForm
    #Folderpath = "./bnal_data/*"
    
    
    
    files = glob.glob(Folderpath)
    print(Folderpath)
    
    
    CX=[]
    for file in files:
        path_load = file
        basename = os.path.splitext(os.path.basename(path_load))[0]
        try:
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
            print(f_name)
            if cc ==0:
                    AX=[]
                    #print(cc,': skipping',basename)
            else:
                AX = sig_anal.apick(file)
            
            CX.extend(AX)
        except ValueError as error:
            print(error)
    
    #保存するパスを作成する
    save_path = ('a_data/'+SamplePath + '_' + TargetPath +'_'+'r.npy')
    np.save(save_path, CX)
        
        
            
if __name__ =='__main__':
    svr = 'Rackstation'
    sfd = 'analysis'
    ex = 'Chirality_MX'
    ExPath = '//' + svr + '/' + sfd + '/' + ex + '/'
    Samples =['4G_L_N_D_WTM','4L_K_D_AIE']
    print(Samples)
    for sample in Samples:
        XX=fe_xx(ExPath, sample)
    print('end')
    #'3G_L_M_D_WTN','4G_L_N_D_WTM','4L_K_D_AIE','