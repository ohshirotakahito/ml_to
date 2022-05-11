# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:36:58 2021

@author: ohshi
"""
import glob
import nptdms
from nptdms import TdmsFile
from nptdms import tdms
import numpy as np

#Folderpath = '\\\\Rackstation\\analysis\\P53_RKO\\P53_RKOR\\P53_RKOR_10k_Sample\\*'
Folderpath = '//Rackstation/analysis/P53_RKO/P53_RKOR/P53_RKOR_10k_Sample/T/BNAL@P53down/*'
files = glob.glob(Folderpath)

target0 = '.tdms_index'
RL=[]
for file in files:
    f = str(file)
    target0 = '.tdms_index'
    idx0 = target0 in f
    if not idx0 == True:
        RL.append(file)

files = RL