#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import os

import numpy as np


class aosEstimator(object):

    def __init__(self, paramFile, icomp, izn3, debugLevel):
        self.filename = os.path.join('data/', (paramFile + '.esti'))
        fid = open(self.filename)
        iscomment = False
        for line in fid:
            line = line.strip()
            if (line.startswith('###')):
                iscomment = ~iscomment
            if (not(line.startswith('#')) and
                    (not iscomment) and len(line) > 0):
                if (line.startswith('estimator_strategy')):
                    self.strategy = line.split()[1]
                elif (line.startswith('senMFile')):
                    self.senMFile = line.split()[1]
                elif (line.startswith('n_bending_M1M3')):
                    self.nB13Max = int(line.split()[1])
                elif (line.startswith('n_bending_M2')):
                    self.nB2Max = int(line.split()[1])
                elif (line.startswith('znmax')):
                    self.znMax = int(line.split()[1])
                elif (line.startswith('normalize_A')):
                    self.normalizeA = bool(int(line.split()[1]))
                elif (line.startswith('n_singular_inf')):
                    self.nSingularInf = int(line.split()[1])
                elif (line.startswith('icomp')):
                    if (icomp is None):
                        self.icomp = int(line.split()[1])
                    else:
                        self.icomp = icomp
                    arrayType = 'icomp'
                    arrayCount = 0
                elif (line.startswith('izn3')):
                    if (izn3 is None):
                        self.izn3 = int(line.split()[1])
                    else:
                        self.izn3 = izn3
                    arrayType = 'izn3'
                    arrayCount = 0
                else:
                    line1 = line.replace('1', '1 ')
                    line1 = line1.replace('0', '0 ')
                    if (line1[0].isdigit()):
                        arrayCount = arrayCount + 1
                    if (arrayType == 'icomp' and arrayCount == self.icomp):
                        self.compIdx = np.fromstring(
                            line1, dtype=bool, sep=' ')
                        arrayCount = 0
                        arrayType = ''
                    elif (arrayType == 'izn3' and arrayCount == self.izn3):
                        self.zn3Idx = np.fromstring(line1, dtype=bool, sep=' ')
                        arrayCount = 0
                        arrayType = ''

        fid.close()

        self.zn3Max = self.znMax - 3
        self.ndofA = self.nB13Max + self.nB2Max + 10

        if debugLevel >= 1:
            print('Using senM file: %s' % self.senMFile)
        self.senM = np.loadtxt(os.path.join('data/', self.senMFile))
        self.senM = self.senM.reshape((-1, self.zn3Max, self.ndofA))
        self.senM = self.senM[:, :, np.concatenate(
            (range(10 + self.nB13Max),
             range(10 + self.nB13Max, 10 + self.nB13Max + self.nB2Max)))]
        if (debugLevel >= 3):
            print(self.strategy)
            print(self.senM.shape)

        # A is just for the 4 corners
        self.A = self.senM[-4:, :, :].reshape((-1, self.ndofA))
        self.zn3IdxAx4 = np.repeat(self.zn3Idx, 4)
        self.Ause = self.A[np.ix_(self.zn3IdxAx4, self.compIdx)]
        if debugLevel >= 3:
            print('---checking estimator related:')
            print(self.compIdx)
            print(self.zn3Idx)
            print(self.zn3Max)
            print(self.Ause.shape)
            print(self.Ause[21, -1])
            print(self.normalizeA)
            
        self.Anorm = self.Ause 
        if (debugLevel >= 3):
            print('---checking Anorm (actually Ause):')
            print(self.Anorm[:5, :5])
            print(self.Ause[:5, :5])
        if self.strategy == 'pinv':
            self.Ainv = pinv_truncate(self.Anorm, self.nSingularInf)
            
    def normA(self, ctrl):
        dofUnitMat = np.repeat(ctrl.Authority.reshape((1,-1)), self.Ause.shape[0] ,axis=0)

        self.Anorm = self.Ause / dofUnitMat
        self.Ainv = pinv_truncate(self.Anorm, self.nSingularInf)
        
        
    def estimate(self, state, wfs, ctrl, sensoroff):
        if sensoroff:
            aa = np.loadtxt(state.zFile_m1)
            self.yfinal = aa[-4:, 3:self.znMax].reshape((-1, 1))
            mu = np.zeros(self.zn3Max*4)
            np.random.seed(state.obsID)
            self.yfinal += np.random.multivariate_normal(mu,wfs.covM).reshape(-1,1)

        self.yfinal -= wfs.intrinsic4c

        # subtract y2c
        aa = np.loadtxt(os.path.join('data/', ctrl.y2File))
        self.y2c = aa[-4:, 0:self.znMax-3].reshape((-1, 1))
        
        self.xhat = np.zeros(self.ndofA)
        self.xhat[self.compIdx] = self.Ainv.dot(self.yfinal[self.zn3IdxAx4]-self.y2c)
        self.yresi = self.yfinal.copy()
        self.yresi -= self.y2c
        self.yresi += np.reshape(
            self.Anorm.dot(-self.xhat[self.compIdx]), (-1, 1))

def pinv_truncate(A, n):
    Ua, Sa, VaT = np.linalg.svd(A)
    siginv = 1 / Sa
    if n > 1:
        siginv[-n:] = 0
    Sainv = np.diag(siginv)
    Sainv = np.concatenate(
        (Sainv, np.zeros(
            (VaT.shape[0], Ua.shape[0] - Sainv.shape[1]))),
        axis=1)
    Ainv = VaT.T.dot(Sainv).dot(Ua.T)
    return Ainv
    
