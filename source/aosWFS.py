#!/usr/bin/env python
##
# @authors: Bo Xin 
# @       Large Synoptic Survey Telescope

import os
import numpy as np

from cwfsAlgo import cwfsAlgo
from cwfsInstru import cwfsInstru

class aosWFS(object):

    def __init__(self,cwfsDir,instruFile,algoFile,
                 imgSizeinPix,wavelength,debugLevel):

        aosDir=os.getcwd()
        self.cwfsDir=cwfsDir
        os.chdir(cwfsDir)
        self.inst = cwfsInstru(instruFile, imgSizeinPix)
        self.algo = cwfsAlgo(algoFile, self.inst, debugLevel)
        os.chdir(aosDir)
        self.znwcs=self.algo.numTerms
        self.znwcs3=self.znwcs-3
        self.myZn=np.zeros((self.znwcs3*4,2))
        self.phosimZn=np.zeros((self.znwcs3*4,2))
        intrinsic35=np.loadtxt('data/intrinsic_zn.txt');
        intrinsic35=intrinsic35*wavelength
        self.intrinsic4c=intrinsic35[
            -4:,3:self.algo.numTerms].reshape((-1,1))
        
        if debugLevel >=3:
            print('znwcs3=%d'%self.znwcs3)
            print(self.intrinsic4c.shape)
            print(self.intrinsic4c[:5])
