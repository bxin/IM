#!/usr/bin/env python
##
# @authors: Bo Xin 
# @       Large Synoptic Survey Telescope

import os
import numpy as np
from scipy.linalg import block_diag

class aosController(object):

    def __init__(self,paramFile, esti, metr, M1M3, M2, wavelength, debugLevel):

        self.filename = os.path.join('data/', (paramFile + '.ctrl'))
        fid = open(self.filename)
        iscomment = False
        for line in fid:
            line = line.strip()
            if (line.startswith('###')):
                iscomment = ~iscomment
            if (not(line.startswith('#')) and
                    (not iscomment) and len(line) > 0):
                if (line.startswith('control_strategy')):
                    self.strategy = line.split()[1]
                elif (line.startswith('shift_gear')):
                    self.shiftGear = bool(int(line.split()[1]))
                elif (line.startswith('M1M3_actuator_penalty')):
                    self.rhoM13 = float(line.split()[1])
                elif (line.startswith('M2_actuator_penalty')):
                    self.rhoM2 = float(line.split()[1])
                elif (line.startswith('Motion_penalty')):
                    self.rho = float(line.split()[1])

        fid.close()
        if debugLevel>=1:
            print('control strategy: %s'%self.strategy)
        if (self.strategy == 'no'):
            mF=np.identity(esti.Ause.shape[0])
        else:
            # use rms^2 as diagnal
            aa = M1M3.force[:,esti.compIdx[10:10+esti.nB13Max]]
            mHM13 = np.diag(np.mean(np.square(aa),axis=0))
            aa = M2.force[:,esti.compIdx[
                10+esti.nB13Max:10+esti.nB13Max+esti.nB2Max]]
            mHM2 = np.diag(np.mean(np.square(aa),axis=0))
            # the block for the rigid body DOF (r for rigid)
            mHr = np.identity(np.sum(esti.compIdx[:10]))
            mH = block_diag(mHr, self.rhoM13**2*mHM13, self.rhoM2**2*mHM2)

            if (self.strategy == 'optiPSSN'):
                #wavelength below in um,b/c output of A in um    
                CCmat=np.diag(metr.pssnAlpha)*(2*np.pi/wavelength)**2
                self.mQ=np.zeros((esti.Ause.shape[1],esti.Ause.shape[1]))
                for iField in range(metr.nField):
                    aa=esti.senM[iField,:,:]
                    Afield=aa[np.ix_(esti.zn3Idx,esti.compIdx)]
                    mQf = Afield.T.dot(CCmat).dot(Afield)
                    self.mQ = self.mQ + metr.w[iField] * mQf
                self.mF = np.linalg.pinv(self.mQ+self.rho**2*mH)
                
                if debugLevel>=3:
                    print(self.mQ[0,0])
                    print(self.mQ[0,9])
                    
    def showZnBeforeAfter(self):
        idxCsubpC=[6, 5, 7, 8]
