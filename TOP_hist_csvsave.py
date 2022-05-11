# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 16:56:32 2021

@author: ohshi
"""

import numpy as np
import csv
import pandas as pd

paths=['OM1-01_10k_Sample_BNAL@FTarget01_b.npy',
      'OM2-01_10k_Sample_BNAL@FTarget01_b.npy',
      'OM3-01_10k_Sample_BNAL@FTarget01_b.npy',
       'OM4-01_10k_Sample_BNAL@FTarget01_b.npy']
base1 = 6
base2 = 16

for path in paths:
    r_path='data/'+path
    print(r_path)
    data = np.load(r_path, allow_pickle=True)
    AX=[]
    for datum in data:
        dd = datum.split(',')
        e0 = int(dd[0].replace('[', ''))
        e2 = float(dd[2])
        if e0 == base1 or base2:
            AX.append(e2)
np.savetxt('csv/RKO1_T.csv', AX, delimiter=",", fmt="%.5f")