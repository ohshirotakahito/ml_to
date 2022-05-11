# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 15:01:05 2021

@author: ohshi
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 11:54:07 2021

@author: ohshi
"""
import selenium
from time import sleep
import pandas as pad
import datetime
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

##起動するブラウザーを選択して起動するまで
#browser = webdriver.Chrome(r'C:\Users\ohshi\Desktop\chromedriver_win32\chromedriver.exe')

def get_web_driver():
#ここで使用環境にあったWebDriverをインストール
    driver=webdriver.Chrome(ChromeDriverManager().install()) 
    return driver

browser =get_web_driver()

browser.get('https://www-kaken.jsps.go.jp/kaken1/shinsei/logon.do?4c2bf78e=cqDtkdK5D64u8rri8y9Pl6iZ8d6TErzLHbSN')

##入力フォームをさがして，値を入れるまで（１）
elem_name=browser.find_element_by_name('userid')
#print(elem_name)
elem_name.send_keys('ohshirotakahito')

##入力フォームをさがして，値を入れるまで（１）
elem_name=browser.find_element_by_name('password')
#print(elem_name)
elem_name.send_keys('Showshow@taka77')

##xpathをもちいてクリックするまで
elem_login=browser.find_element_by_xpath('/html/body/div/table[2]/tbody/tr/td[2]/table[2]/tbody/tr/td[3]/form/table/tbody/tr[5]/td[4]/input')
#time.sleep(1)
elem_login.click()


# =============================================================================
# ##起動するブラウザーが起動するアドレス
# browser.get('https://ou-idp.auth.osaka-u.ac.jp/idp/sso_redirect?SAMLRequest=fVJbT4MwFH73V5C%2BQzs2JGsYyXQxLplKBvrgy1JKlSq02FOM%2B%2Fdy8TJNtj6efvlu50TA6qqhy9aWaiveWgHW%2BagrBXT4WKDWKKoZSKCK1QKo5TRd3myo7xHaGG011xVylgDCWKnVpVbQ1sKkwrxLLu63mwUqrW2AYlzvPQ3slbmtx7j30uC0lHmuK2FLD0DjntfHyV2aIWfVGZGK9ZS%2FBLp1ZdF4rDP7j6kb445iZ0QhjeAWOevVAu0mQcCC2TzMRcF5MZmGZDrnM5ZPZ6FfFDnpYACtWCuwTNkF8ok%2Fccm5S%2FyM%2BDQIKAkfkZN8xbyQqpDq%2BXQn%2BQgCep1liTuGeRAGhiAdAMVnzvEX9a3TwZQ52MNpSfZdPoqPVg0%2FVUf4QOOYmWg8i9tOa71KdCX53rnSpmb2tJV%2BIgv3aYBSa5gCKZRFOB5l%2F15a%2FAk%3D&RelayState=https%3A%2F%2Fmy.osaka-u.ac.jp%2F&SigAlg=http%3A%2F%2Fwww.w3.org%2F2001%2F04%2Fxmldsig-more%23rsa-sha256&Signature=RAH6BWXnPFyF3%2FGWPAZkJYoZKsTpJRpGA%2F24Fl6ctO%2BJF5fxAMA1r%2BV8gJv9NP2sSFSg8FRi6S7RX9hHgjHQ%2BDJGLObSe4libvzaYE8HZLoCcI0hhzo%2BzDHhwu1W4L4ti5gLN7wGVTICjcQaGazbBJzfEpAjSQSA7A%2BLLoKmfx%2FtwQ9Dyzzgug079La9nHeKQFBEMYw8%2FLMt%2FJD1S1jv23KHkGguNDYCpjrGBXtGAgIo2qqcBMdISaugbvaXVpKpYxE2o9075PEj%2BoosRAdRgsinUMaqidifLQsSlUM%2B3oP0qh1UmvUSJszyfigiwelfO%2Fa3vQDbEkgjik5hZtfLDA%3D%3D')
# 
# ##入力フォームをさがして，値を入れるまで（１）
# elem_name=browser.find_element_by_name('USER_ID')
# #print(elem_name)
# elem_name.send_keys('u297333j')
# 
# ##入力フォームをさがして，値を入れるまで（１）
# elem_name=browser.find_element_by_name('USER_PASSWORD')
# #print(elem_name)
# elem_name.send_keys('Showshow@taka77')
# 
# ##xpathをもちいてクリックするまで
# elem_login=browser.find_element_by_xpath('/html/body/table/tbody/tr[3]/td/table/tbody/tr[5]/td/table/tbody/tr/td[2]/div/input')
# #time.sleep(1)
# elem_login.click()
# 
# 
# ##xpathをもちいてクリックするまで
# elem_login=browser.find_element_by_xpath('//*[@id="my-handai-top-right-column"]/div/div[1]/div/div[3]/p[2]/a')
# #time.sleep(1)
# elem_login.click()
# 
# ##xpathをもちいてクリックするまで
# elem_login=browser.find_element_by_xpath('//*[@id="PS_siteLinkList_main"]/table/tbody/tr/td[1]/dl/dt[1]/a')
# #time.sleep(1)
# =============================================================================
elem_login.click()