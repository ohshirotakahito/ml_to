# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:39:10 2021

@author: ohshi
"""

def a_abb(x): 
    amino = {'Ala':'A','Arg':'R','Asn':'N',
     'Asp':'D','Cys':'C','Gln':'Q',
     'Glu':'E','Gly':'G','His':'H',
     'Ile':'I','Leu':'L','Lys':'K',
     'Met':'M','Phe':'F','Pro':'P',
     'Ser':'S','Thr':'T','Trp':'W',
     'Tyr':'Y','Val':'V',}
    
    x_ans = amino.get(x)
    return x_ans

def a_abb2(x):     
    am = {'A': 'Ala', 'R': 'Arg', 'D': 'Asp',
          'C': 'Cys', 'Q': 'Gln', 'E': 'Glu',
          'G': 'Gly', 'H': 'His', 'I': 'Ile',
          'L': 'Leu', 'K': 'Lys', 'M': 'Met',
          'F': 'Phe', 'P': 'Pro', 'S': 'Ser',
          'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr',
          'V': 'Val', 'N': 'Asn'}
    y_ans = am.get(x)
    return y_ans

if __name__ == '__main__':
    x = 'Trp'
    p = a_abb(x)
    z = a_abb2(x)
    print(p,z)





