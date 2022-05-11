# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 15:49:57 2021

@author: ohshi
"""
import time
 
t0 = time.time()                # 計測開始時間を取得

import itertools

s = str(input())
t = str(input())

#print(s)
#print(t)

S=[]
for i in range(len(s)):
    si = str(s[i])
    S.append(si)

#print(S)

Ans=[]
for pair in itertools.permutations(S, len(S)):
    a_i=''.join(pair)
    if str(a_i) != str(s):
        Ans.append(a_i)

ax=0
for ans in Ans:
    if str(ans) == t:
        ax=+1
if ax==0:
    print('NO')
else:
    print('YES')

t1 = time.time()                # 計測終了時間
elapsed_time = float(t1 - t0)   # 経過時間
print(elapsed_time)   