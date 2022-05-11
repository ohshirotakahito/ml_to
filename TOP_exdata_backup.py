# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 16:57:05 2021

@author: ohshi
"""

import os
import glob
import shutil


folder = 'backup_05'
t_folder = 'data/AN1'

ServerPath = '//SQserver'
KeyFolderPath= folder

#対象となるフォルダパスを作成
Folderpath = ServerPath + KeyFolderPath


Transfer_folders = os.listdir(path = Folderpath)

print(Transfer_folders)