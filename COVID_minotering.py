# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 14:49:11 2021

@author: ohshi
"""
from bs4 import BeautifulSoup
from urllib import request
import numpy as np

url = 'https://www.kantei.go.jp/jp/headline/kansensho/vaccine.html'
response = request.urlopen(url)
soup = BeautifulSoup(response,'html.parser')
response.close()

soup = str(soup)

a = soup.find('総接種回数')
b = soup.find('</font>回')
c=soup.find('日')
d=soup.find('令和')

soup_y = soup[a:b]

e = soup_y.split('">')

f = e[1].split(';')
t=int(f[0].replace(',', ''))

soup_x = soup[d:c+1]

jap_pop = int(123334135)

xx=t/jap_pop
wariai = xx*100
z=str(round(wariai,2))

print('更新日:'+soup_x)
print('これまでのワクチン総接種数:'+f[0]+' 回')
print('ここまでの接種割合： '+ z+' %')