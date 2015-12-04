#!/usr/bin/env python
##
# @authors: Bo Xin 
# @       Large Synoptic Survey Telescope

import os
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

class aosController(object):

    def __init__(self,paramFile, esti, metr, M1M3, M2, wavelength, gain,
                 debugLevel):

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
        self.gain = gain
        
        if (self.strategy == 'null'):
            self.mF=np.identity(esti.Ause.shape[1])
        else:
            # use rms^2 as diagnal
            aa = M1M3.force[:, :esti.nB13Max]
            aa = aa[:,esti.compIdx[10:10+esti.nB13Max]]
            mHM13 = np.diag(np.mean(np.square(aa),axis=0))
            aa = M2.force[:, :esti.nB2Max]
            aa = aa[:,esti.compIdx[
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

    def getMotions(self, esti, state):
        self.uk = - self.gain * self.mF.dot(esti.xhat)

        self.drawControlPanel(esti, state)
        
    def drawControlPanel(self, esti, state):

        fig = plt.figure(figsize=(15, 10))

        # rigid body motions
        axm2rig = plt.subplot2grid((4,4),(0,0))
        axm2rot = plt.subplot2grid((4,4),(0,1))
        axcamrig = plt.subplot2grid((4,4),(0,2))
        axcamrot = plt.subplot2grid((4,4),(0,3))

        myxticks = [1,2,3]
        myxticklabels = ['%d'%(myxticks[i]) for i in np.arange(len(myxticks))]
        axm2rig.plot(myxticks, self.uk[[(i-1) for i in myxticks]],'ro',ms=8)
        axm2rig.set_xticks(myxticks)
        axm2rig.set_xticklabels(myxticklabels)
        axm2rig.grid()
        axm2rig.annotate('M2 dz,dx,dy', xy=(0.3, 0.4),
                         xycoords='axes fraction', fontsize=16)
        axm2rig.set_ylabel('um')
        axm2rig.set_xlim(np.min(myxticks)-0.5, np.max(myxticks)+0.5)
        
        myxticks = [4,5]
        myxticklabels = ['%d'%(myxticks[i]) for i in np.arange(len(myxticks))]
        axm2rot.plot(myxticks, self.uk[[(i-1) for i in myxticks]],'ro',ms=8)
        axm2rot.set_xticks(myxticks)
        axm2rot.set_xticklabels(myxticklabels)
        axm2rot.grid()
        axm2rot.annotate('M2 rx,ry', xy=(0.3, 0.4),
                         xycoords='axes fraction', fontsize=16)
        axm2rot.set_ylabel('arcsec')
        axm2rot.set_xlim(np.min(myxticks)-0.5, np.max(myxticks)+0.5)
        
        myxticks = [6,7,8]
        myxticklabels = ['%d'%(myxticks[i]) for i in np.arange(len(myxticks))]
        axcamrig.plot(myxticks, self.uk[[(i-1) for i in myxticks]],'ro',ms=8)
        axcamrig.set_xticks(myxticks)
        axcamrig.set_xticklabels(myxticklabels)
        axcamrig.grid()
        axcamrig.annotate('Cam dz,dx,dy', xy=(0.3, 0.4),
                         xycoords='axes fraction', fontsize=16)
        axcamrig.set_ylabel('um')
        axcamrig.set_xlim(np.min(myxticks)-0.5, np.max(myxticks)+0.5)
        
        myxticks = [9,10]
        myxticklabels = ['%d'%(myxticks[i]) for i in np.arange(len(myxticks))]
        axcamrot.plot(myxticks, self.uk[[(i-1) for i in myxticks]],'ro',ms=8)
        axcamrot.set_xticks(myxticks)
        axcamrot.set_xticklabels(myxticklabels)
        axcamrot.grid()
        axcamrot.annotate('Cam rx,ry', xy=(0.3, 0.4),
                         xycoords='axes fraction', fontsize=16)
        axcamrot.set_ylabel('arcsec')
        axcamrot.set_xlim(np.min(myxticks)-0.5, np.max(myxticks)+0.5)
        
        # m13 and m2 bending
        axm13 = plt.subplot2grid((4,4),(1,0), colspan =2 )
        axm2 = plt.subplot2grid((4,4),(1,2), colspan =2 )

        myxticks = range(1, esti.nB13Max+1)
        myxticklabels = ['%d'%(myxticks[i]) for i in np.arange(len(myxticks))]
        axm13.plot(myxticks, self.uk[[(i-1+10) for i in myxticks]],'ro',ms=8)
        axm13.set_xticks(myxticks)
        axm13.set_xticklabels(myxticklabels)
        axm13.grid()
        axm13.annotate('M1M3 bending', xy=(0.3, 0.4),
                         xycoords='axes fraction', fontsize=16)
        axm13.set_ylabel('um')
        axm13.set_xlim(np.min(myxticks)-0.5, np.max(myxticks)+0.5)
        
        myxticks = range(1, esti.nB2Max+1)
        myxticklabels = ['%d'%(myxticks[i]) for i in np.arange(len(myxticks))]
        axm2.plot(myxticks, self.uk[[(
            i-1+10+esti.nB13Max) for i in myxticks]],'ro',ms=8)
        axm2.set_xticks(myxticks)
        axm2.set_xticklabels(myxticklabels)
        axm2.grid()
        axm2.annotate('M2 bending', xy=(0.3, 0.4),
                         xycoords='axes fraction', fontsize=16)
        axm2.set_ylabel('um')
        axm2.set_xlim(np.min(myxticks)-0.5, np.max(myxticks)+0.5)
                
        # OPD zernikes before and after the FULL correction
        # it goes like
        #  2 1
        #  3 4
        axz1 = plt.subplot2grid((4,4),(2,2), colspan =2 )
        axz2 = plt.subplot2grid((4,4),(2,0), colspan =2 )
        axz3 = plt.subplot2grid((4,4),(3,0), colspan =2 )
        axz4 = plt.subplot2grid((4,4),(3,2), colspan =2 )

        z4up = range(4, esti.znMax+1)
        axz1.plot(z4up, esti.yfinal[:esti.zn3Max],
                  label='iter %d'%(state.iIter-1),
                  marker='*', color='b', markersize=10)
        axz1.plot(z4up, esti.yresi[:esti.zn3Max],
                  label='if full correction applied',
                  marker='*', color='r', markersize=10)
        axz1.grid()
        axz1.annotate('Zernikes R44', xy=(0.3, 0.4),
                         xycoords='axes fraction', fontsize=16)
        axz1.set_ylabel('um')
        axz1.legend(loc="best", shadow=True, fancybox=True)
        axz1.set_xlim(np.min(z4up)-0.5, np.max(z4up)+0.5)
        
        axz2.plot(z4up, esti.yfinal[esti.zn3Max:2*esti.zn3Max],
                    marker='*', color='b', markersize=10)
        axz2.plot(z4up, esti.yresi[esti.zn3Max:2*esti.zn3Max],
                  marker='*', color='r', markersize=10)                  
        axz2.grid()
        axz2.annotate('Zernikes R40', xy=(0.3, 0.4),
                         xycoords='axes fraction', fontsize=16)
        axz2.set_ylabel('um')
        axz2.set_xlim(np.min(z4up)-0.5, np.max(z4up)+0.5)
                        
        axz3.plot(z4up, esti.yfinal[2*esti.zn3Max:3*esti.zn3Max],
                marker='*', color='b', markersize=10)
        axz3.plot(z4up, esti.yresi[2*esti.zn3Max:3*esti.zn3Max],
                marker='*', color='r', markersize=10)
        axz3.grid()
        axz3.annotate('Zernikes R00', xy=(0.3, 0.4),
                         xycoords='axes fraction', fontsize=16)
        axz3.set_ylabel('um')
        axz3.set_xlim(np.min(z4up)-0.5, np.max(z4up)+0.5)
                        
        axz4.plot(z4up, esti.yfinal[3*esti.zn3Max:4*esti.zn3Max],
                  marker='*', color='b', markersize=10)                  
        axz4.plot(z4up, esti.yresi[3*esti.zn3Max:4*esti.zn3Max],
                  marker='*', color='r', markersize=10)                  
        axz4.grid()
        axz4.annotate('Zernikes R04', xy=(0.3, 0.4),
                         xycoords='axes fraction', fontsize=16)
        axz4.set_ylabel('um')
        axz4.set_xlim(np.min(z4up)-0.5, np.max(z4up)+0.5)
                
        plt.tight_layout()
        
        # plt.show()
        pngFile = '%s/sim%d_iter%d_ctrl.png'%(
                state.pertDir, state.iSim,state.iIter)        
        plt.savefig(pngFile,bbox_inches='tight')
        
