#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import os
import glob

import numpy as np
from aosTeleState import aosTeleState

class aosEstimator(object):

    def __init__(self, instruFile, paramFile, wfs, icomp, izn3, debugLevel):
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
                elif (line.startswith('range_of_motion')):
                    self.fmotion = float(line.split()[1])
                elif (line.startswith('regularization')):
                    self.reguMu = float(line.split()[1])
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
                        self.dofIdx = np.fromstring(
                            line1, dtype=bool, sep=' ')
                        arrayCount = 0
                        arrayType = ''
                    elif (arrayType == 'izn3' and arrayCount == self.izn3):
                        self.zn3Idx = np.fromstring(line1, dtype=bool, sep=' ')
                        arrayCount = 0
                        arrayType = ''

        fid.close()

        aa = instruFile
        if aa[-2:].isdigit():
            aa = aa[:-2]
        src = glob.glob('data/%s/senM*txt' % (aa))
        self.senMFile = src[0]
        self.zn3Max = self.znMax - 3
        self.ndofA = self.nB13Max + self.nB2Max + 10

        if debugLevel >= 1:
            print('Using senM file: %s' % self.senMFile)
        self.senM = np.loadtxt(self.senMFile)
        self.senM = self.senM.reshape((-1, self.zn3Max, self.ndofA))
        self.senM = self.senM[:, :, np.concatenate(
            (range(10 + self.nB13Max),
             range(10 + self.nB13Max, 10 + self.nB13Max + self.nB2Max)))]
        if (debugLevel >= 3):
            print(self.strategy)
            print(self.senM.shape)

        # A is just for the 4 corners
        self.A = self.senM[-wfs.nWFS:, :, :].reshape((-1, self.ndofA))
        self.zn3IdxAx4 = np.repeat(self.zn3Idx, wfs.nWFS)
        self.Ause = self.A[np.ix_(self.zn3IdxAx4, self.dofIdx)]
        if debugLevel >= 3:
            print('---checking estimator related:')
            print(self.dofIdx)
            print(self.zn3Idx)
            print(self.zn3Max)
            print(self.Ause.shape)
            print(self.Ause[21, -1])
            if self.strategy == 'pinv':
                print(self.normalizeA)

        self.Anorm = self.Ause
        self.xhat = np.zeros(self.ndofA)
        if (debugLevel >= 3):
            print('---checking Anorm (actually Ause):')
            print(self.Anorm[:5, :5])
            print(self.Ause[:5, :5])
        if self.strategy == 'pinv':
            self.Ainv = pinv_truncate(self.Anorm, self.nSingularInf)
        elif self.strategy == 'opti' or self.strategy == 'kalman':
            # empirical estimates (by Doug M.), not used when self.fmotion<0
            aa = [0.5, 2, 2, 0.1, 0.1, 0.5, 2, 2, 0.1, 0.1]
            dX = np.concatenate(
                (aa, 0.01 * np.ones(20), 0.005 * np.ones(20)))**2
            X = np.diag(dX)
            if self.strategy == 'opti':
                self.Ainv = X.dot(self.Anorm.T).dot(
                    np.linalg.pinv(self.Anorm.dot(X).dot(self.Anorm.T) + wfs.covM))
            elif self.strategy == 'kalman':
                self.P = np.zeros((self.ndofA, self.ndofA))
                self.Q = X
                self.R = wfs.covM*100
                
        elif self.strategy == 'crude_opti':
            self.Ainv = self.Anorm.T.dot(np.linalg.pinv(
                self.Anorm.dot(self.Anorm.T) +
                self.reguMu * np.identity(self.Anorm.shape[0])))

            
    def normA(self, ctrl):
        self.dofUnit = 1 / ctrl.Authority
        dofUnitMat = np.repeat(self.dofUnit.reshape(
            (1, -1)), self.Ause.shape[0], axis=0)

        self.Anorm = self.Ause / dofUnitMat
        self.Ainv = pinv_truncate(self.Anorm, self.nSingularInf)

    def optiAinv(self, ctrl, wfs):
        dX = (ctrl.range * self.fmotion)**2
        X = np.diag(dX)
        self.Ainv = X.dot(self.Anorm.T).dot(
            np.linalg.pinv(self.Anorm.dot(X).dot(self.Anorm.T) + wfs.covM))

    def estimate(self, state, wfs, ctrl, sensor):
        if sensor == 'ideal' or sensor == 'covM':
            bb = np.zeros((wfs.znwcs, state.nOPDw))
            if state.nOPDw == 1:
                aa = np.loadtxt(state.zTrueFile_m1)
                self.yfinal = aa[-wfs.nWFS:, 3:self.znMax].reshape((-1, 1))
            else:
                for irun in range(state.nOPDw):
                    aa = np.loadtxt(state.zTrueFile_m1.replace('.zer','_w%d.zer'%irun))
                    bb[:, irun] = aa[-wfs.nWFS:, 3:self.znMax].reshape((-1, 1))
                self.yfinal = np.sum(aosTeleState.GQwt * bb)
            if sensor == 'covM':
                mu = np.zeros(self.zn3Max * 4)
                np.random.seed(state.obsID)
                self.yfinal += np.random.multivariate_normal(
                    mu, wfs.covM).reshape(-1, 1)
        else:
            aa = np.loadtxt(wfs.zFile_m1[0]) #[0] for exp No. 0
            self.yfinal = aa[:, :self.zn3Max].reshape((-1, 1))

        self.yfinal -= wfs.intrinsicWFS

        # subtract y2c
        aa = np.loadtxt(ctrl.y2File)
        self.y2c = aa[-wfs.nWFS:, 0:self.znMax - 3].reshape((-1, 1))

        z_k = self.yfinal[self.zn3IdxAx4] - self.y2c
        if self.strategy == 'kalman':
            # the input to each iteration (in addition to Q and R) :
            #         self.xhat[:, state.iIter - 1]
            #         self.P[:, :, state.iIter - 1]

            if state.iIter>1: #for iIter1, iter0 initialized by estimator
                Kalman_xhat_km1_File = '%s/iter%d/sim%d_iter%d_Kalman_xhat.txt' % (
                    self.pertDir, self.iIter-1, self.iSim, self.iIter-1)
                Kalman_P_km1_File = '%s/iter%d/sim%d_iter%d_Kalman_P.txt' % (
                    self.pertDir, self.iIter-1, self.iSim, self.iIter-1)
                self.xhat = np.loadtxt(Kalman_xhat_km1_File)
                self.P = np.loadtxt(Kalman_P_km1_File)
            # time update
            xhatminus_k = self.xhat
            Pminus_k = self.P + self.Q
            # measurement update
            K_k = Pminus_k.dot(self.Anorm.T).dot(
                pinv_truncate(
                    self.Anorm.dot(Pminus_k).dot(self.Anorm.T) + self.R, 5))
            self.xhat[self.dofIdx] = self.xhat[self.dofIdx] + \
              K_k.dot(z_k - np.reshape(self.Anorm.dot(xhatminus_k),(-1,1)))
            self.P[np.ix_(self.dofIdx, self.dofIdx)] = \
              (1-K_k.dot(self.Anorm)).dot(Pminus_k)
              
            Kalman_xhat_k_File = '%s/iter%d/sim%d_iter%d_Kalman_xhat.txt' % (
                state.pertDir, state.iIter, state.iSim, state.iIter)
            Kalman_P_k_File = '%s/iter%d/sim%d_iter%d_Kalman_P.txt' % (
                state.pertDir, state.iIter, state.iSim, state.iIter)
            np.savetxt(Kalman_xhat_k_File, self.xhat)
            np.savetxt(Kalman_P_k_File, self.P)
        else:
            self.xhat[self.dofIdx] = np.reshape(self.Ainv.dot(z_k), [-1])
            if self.strategy == 'pinv' and self.normalizeA:
                self.xhat[self.dofIdx] = self.xhat[self.dofIdx] / self.dofUnit
        self.yresi = self.yfinal.copy()
        self.yresi -= self.y2c
        self.yresi += np.reshape(
            self.Ause.dot(-self.xhat[self.dofIdx]), (-1, 1))


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
