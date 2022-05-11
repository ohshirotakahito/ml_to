# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 11:53:38 2021

@author: ohshi
"""
from pulearn import ElkanotoPuClassifier
from sklearn.svm import SVC
import numpy as np
from numpy.random import*
import matplotlib.pyplot as plt
from scipy.stats import norm


mu =[0,0]
sigma =[[30,20],[20,50]]
values = multivariate_normal(mu,sigma,100)
values0 = multivariate_normal(mu,sigma,100)

x, y = values[:,0], values[:,1]

x0, y0 = values[:,0], values[:,1]


mu1 =[10,12]
sigma1 =[[20,10],[25,80]]
values1 = multivariate_normal(mu1,sigma1,100)

x1, y1 = values1[:,0], values1[:,1]

x2, y2 = np.concatenate([x0,x1],0), np.concatenate([y0,y1],0)

svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
pu_estimator.fit(X, y)