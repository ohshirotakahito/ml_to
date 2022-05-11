# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:42:04 2022

@author: ohshi
"""
import os

ServerPath = '//Rackstation/analysis/'
ex = 'PhiXDNA (TO)'
path = ServerPath + ex

folders = os.listdir(path)
print(folders)