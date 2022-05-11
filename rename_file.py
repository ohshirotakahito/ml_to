# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:36:40 2021

@author: ohshi
"""
import AA_e

ex = '200c(OM)'
sample = '200c_native_C9_000' 
target = 'FTarget01'
files = AA_e.Analextract(ex, sample, target)

#print(files)

files1 = []
for file in files:
    _ = file.split('/')
    __ =_[-1].split('ANAL\A@@')
    ___ = __[0].split(' D_')

    
    files.append('A@@'+___[0])

print()                