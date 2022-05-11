# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 10:20:36 2021

@author: ohshi
"""
class TestClass2:
    val = []
    def __init__(self):
        print("init:" + str(self.val))
        # 初期化
        self.val.append(1)
        self.val.append(2)
        self.val.append(3)

    def test_method1(self):
        print("test_method2:" + str(self.val))

class Cat:
    #
    # クラス変数
    # self は不要
    #
    family = '猫'
    
    def say(self):
        print('にゃー')
    
    def growl(self):
        print('ウー')
    
    def __init__(self, name):
        #
        # インスタンス変数
        # self は必要
        #
        self.name = name