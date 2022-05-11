# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:05:26 2021

@author: ohshi
"""

import numpy as np

#Bnalのアセンブリデータを取得する (TOP_assembly.pyでTDMSのBNALファイルを処理後おこなうプロセス)

samples =['KTRNA01']

Target_Original = 'hsa-miR-15a-5p'
TBase = 'N'
t_b = 12
Target_Base =TBase+'_'+ str(t_b)
#修飾配列番号を指定する


Target_Modification_Position = Target_Original+'_'+Target_Base

# =============================================================================
##ターゲット配列番号
# t_bs = [3,5,8,15,18]
# t_b = 8
#アセンブリ領域を選択
# rns =[3,5,8,15,18]
# 
# for rn in rns:
# =============================================================================
    

#アセンブリ領域を選択
rn = 0

#アセンブリ領域の選択肢作成
RN = [(0,0),(0,1),(1,0),(1,1),(1,2),(2,1),(2,2),(3,2),(2,3),(3,3),(4,3),(3,4),(4,4)]

RR = RN[rn]

t_bs = t_b-RR[0]
t_be = t_b+RR[1]

tse = (t_bs,t_be)

print('start',t_bs,'end',t_be)

Sams = []

for sample in samples:
    Sam = (sample+'_10k_Sample_BNAL@'+Target_Original+'_r',
        sample+'_10k_Sample_BNAL@'+Target_Modification_Position+'_r')
    Sams.append(Sam)



for samp in Sams:
    CN=[]
    for sam in samp:
        datum = np.load('data/'+sam+'.npy', allow_pickle=True)
        #Sequence = 'CCTGTAGTCCCAGC'
 
        count = 0
        
        for data in datum:
            data = data.split(',')
            sbn, sen = int(data[3]),int(data[4])
            #print(sbn)
            if sbn <= t_bs and sen >= t_be:
                count = count+1
        CN.append(count)
        print(str(sam)+': ',count, len(datum))
    print('+1',CN,'{:.2f}'.format(CN[1]/(CN[0]+CN[1])*100)+'%')