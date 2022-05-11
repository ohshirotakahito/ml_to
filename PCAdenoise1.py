# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:43:13 2021

c.f.
Yasunari Kusaka et al. J. Phys Chem A 123, 10333 (2019) 

@author: komoto
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

class DataHandling(object):
    def __init__(self,filedir='',savedir=''):
        self.spectral_matrix = []
        self.filedir = filedir
        self.savedir = savedir
        
        if not os.path.exists(self.savedir) and self.savedir :
            os.mkdir(self.savedir)
            
    def load(self):
        files = glob.glob(self.filedir+'*.txt')
        
        self.datapoint = len(np.loadtxt(files[0])[:,1])
        self.spectrum_num = len(files)
        self.xaxis = np.loadtxt(files[0])[:,0]
        
        self.spectral_matrix=np.zeros([self.spectrum_num,self.datapoint])
        
        
        for i,file in enumerate(files):
            self.spectral_matrix[i] = np.loadtxt(file)[:,1]
            
        print('load finished')
        return 1
        

class PCAdenoising(DataHandling):
    def __init__(self,filedir='',savedir='',n_components=10,denoise_num = 5):
        super().__init__(filedir=filedir,savedir=savedir)
        self.load()
        self.n_components=n_components
        
        self.denoise_num = denoise_num
        
    def fit(self):
        self.pca = PCA(n_components=self.n_components)
        self.reconstructed_matrix = self.pca.fit_transform(self.spectral_matrix)
        
        
    def plotEigenValue(self):##Figure 4
        eigenvalues = self.pca.explained_variance_ratio_
        
        plt.plot(np.arange(len(eigenvalues))+1, eigenvalues )
        plt.yscale('log')
        plt.ylabel('Normalized Eigenvalue')
        plt.xlabel('Factor level')
        plt.show()
        
        
    def plotEigenVector(self,margin=0.1):#Figure 5
        temp = 0
        for i in range(self.n_components ):
            spec = self.pca.components_[self.n_components-i-1]
            if np.max(spec) < -1*np.min(spec):
                spec = -spec
            
            plt.plot(self.xaxis, spec+temp)
            temp = np.max(spec)+margin+temp
        plt.xlabel('Chemical shift / ppm')
        plt.yticks([])
        plt.show()
        
    def denoise(self,spec_num=0):
        vec = self.reconstructed_matrix[spec_num]        
        reconstructed_spec = 1*self.pca.mean_# 1*がないとself.pca.mean_が上書きされて間違う
        
        for i in range(self.denoise_num):
            reconstructed_spec += vec[i]*self.pca.components_[i] 
        return reconstructed_spec
        
    
    def denoise_plot(self,spec_num=0,xrange=[],xrange_expand=[],magnification=10):#figure 6
        original_spec = self.spectral_matrix[spec_num]
        reconstructed_spec = self.denoise(spec_num)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        ax1.plot(self.xaxis,original_spec)
        ax2.plot(self.xaxis,reconstructed_spec)
        
        if xrange_expand:
            
            temp= ( xrange_expand[0]<self.xaxis) * (self.xaxis<xrange_expand[1])
            
            x = self.xaxis[temp]
            
            original_y = magnification * original_spec[temp]
            reconstructed_y = magnification * reconstructed_spec[temp]
            
            ax1.plot(x,original_y)
            ax2.plot(x,reconstructed_y)
            
        if xrange:
            ax1.set_xlim(xrange[0],xrange[1])
            ax2.set_xlim(xrange[0],xrange[1])
            
        fig.show()
        
    def test(self):
        cov = np.cov(self.spectral_matrix.T)
        w,v = np.linalg.eigh(cov)
        
        idx = w.argsort()[::-1]
        self.w = w[idx]
        self.v = v[:,idx].T
        
if __name__ == '__main__':
    
    filedir = os.getcwd()+'\\testdata2\\ex2\\'
    
    nr = PCAdenoising(filedir=filedir,n_components=10)
    
    nr.fit()
    nr.plotEigenValue()
    nr.plotEigenVector()
    
    nr.denoise_num=7
    
    for i in range(100):
        nr.denoise_plot(spec_num=i ,xrange=[14,40],xrange_expand=[14,19])
    
    
    
    
    
            
            
            
            
            
            

    



