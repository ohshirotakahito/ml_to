# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:46:04 2021

@author: ohshi
"""

N, X, Y = input().split()

N = int(N)
X = int(X)
Y = int(Y)


for i in range(1, N+1):
    if i % X == 0 and i % Y == 0:
        print('AB')
    elif i % X == 0:
        print('A')
    elif i % Y == 0:
        print('B')
    else:
        print('N')    
