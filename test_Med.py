# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:21:05 2021

@author: ohshi
"""
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import seaborn

original_df = pd.read_csv('data/No-show-Issue-Comma-300k.csv')
print(original_df.head(6))