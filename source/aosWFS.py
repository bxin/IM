#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import os
import glob
import multiprocessing

import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib.pyplot as plt

from cwfsAlgo import cwfsAlgo
from cwfsInstru import cwfsInstru
from cwfsImage import cwfsImage

class aosWFS(object):

    def __init__(self, cwfsDir, instruFile, algoFile,
                 imgSizeinPix, wavelength, debugLevel):
        self.nWFS = 4
        self.nRun = 1
        if instruFile[:6] == 'comcam':
            self.nWFS = 9
            self.nRun = 2
        self.wfsName = ['intra', 'extra']
        self.offset = [-1.5, 1.5] #default offset
        if  (instruFile[:6] == 'comcam' and len(instruFile)==8) or \
          (instruFile[:4] == 'lsst' and len(instruFile)==6):
            aa = float(instruFile[-2:])/10
            self.offset = [-aa, aa]
            
        self.halfChip = ['C1', 'C0'] #C1 is always intra, C0 is extra

        aosDir = os.getcwd()
        self.cwfsDir = cwfsDir
        os.chdir(cwfsDir)
        self.inst = cwfsInstru(instruFile, imgSizeinPix)
        self.algo = cwfsAlgo(algoFile, self.inst, debugLevel)
        os.chdir(aosDir)
        self.znwcs = self.algo.numTerms
        self.znwcs3 = self.znwcs - 3
        self.myZn = np.zeros((self.znwcs3 * self.nWFS, 2))
        self.trueZn = np.zeros((self.znwcs3 * self.nWFS, 2))
        aa = instruFile
        if aa[-2:].isdigit():
            aa = aa[:-2]
        intrinsicAll = np.loadtxt('data/%s/intrinsic_zn.txt'%aa)
        intrinsicAll = intrinsicAll * wavelength
        self.intrinsicWFS = intrinsicAll[
            -self.nWFS:, 3:self.algo.numTerms].reshape((-1, 1))
        self.covM = np.loadtxt('data/covM86.txt') #in unit of nm^2
        if self.nWFS>4:
            nrepeat = np.ceil(self.nWFS/4)
            self.covM = np.tile(self.covM, (nrepeat, nrepeat))
            self.covM = self.covM[:(self.znwcs3 * self.nWFS), :(self.znwcs3 * self.nWFS)]
        self.covM = self.covM*1e-6 #in unit of um^2
        
        if debugLevel >= 3:
            print('znwcs3=%d' % self.znwcs3)
            print(self.intrinsicWFS.shape)
            print(self.intrinsicWFS[:5])

    def preprocess(self, state, metr, debugLevel):
        for iField in range(metr.nFieldp4-self.nWFS, metr.nFieldp4):
            chipStr, px0, py0 = state.fieldXY2Chip(
                metr.fieldXp[iField], metr.fieldYp[iField], debugLevel)
            for ioffset in [0, 1]:
                src = glob.glob('%s/iter%d/*%d*%s*%s*' %
                                (state.imageDir, state.iIter, state.obsID, chipStr, self.halfChip[ioffset]))
                chipFile = src[0]
                IHDU = fits.open(chipFile)
                chipImage = IHDU[0].data
                IHDU.close()

                if state.inst[:4]=='lsst':
                    if ioffset == 0: #intra image, C1, push away from left edge
                        # degree to micron then to pixel
                        px = px0 + 0.020 * 180000 / 10 - chipImage.shape[1] 
                    elif ioffset == 1: #extra image, C0, pull away from right edge
                        px = px0 - 0.020 * 180000 / 10
                elif state.inst[:6]=='comcam':
                    px =  px0
                py = py0.copy()
                
                # psf here is 4 x the size of cwfsStampSize, to get centroid
                psf = chipImage[np.max((0, py - 2 * state.cwfsStampSize)):
                                py + 2 * state.cwfsStampSize,
                                np.max((0, px - 2 * state.cwfsStampSize)):
                                px + 2 * state.cwfsStampSize]
                centroid = ndimage.measurements.center_of_mass(psf)
                offsety = centroid[0] - 2 * state.cwfsStampSize + 1
                offsetx = centroid[1] - 2 * state.cwfsStampSize + 1
                # if the psf above has been cut on px=0 or py=0 side
                if py - 2 * state.cwfsStampSize < 0:
                    offsety -= py - 2 * state.cwfsStampSize
                if px - 2 * state.cwfsStampSize < 0:
                    offsetx -= px - 2 * state.cwfsStampSize

                psf = chipImage[
                    py - state.psfStampSize / 2 + offsety:
                    py + state.psfStampSize / 2 + offsety,
                    px - state.psfStampSize / 2 + offsetx:
                    px + state.psfStampSize / 2 + offsetx]

                if state.inst[:4]=='lsst':
                    # readout of corner raft are identical,
                    # cwfs knows how to handle rotated images
                    # note: rot90 rotates the array,
                    # not the image (as you see in ds9, or Matlab with "axis xy")
                    # that is why we need to flipud and then flip back
                    if iField == metr.nField:
                        psf = np.flipud(np.rot90(np.flipud(psf), 2))
                    elif iField == metr.nField+1:
                        psf = np.flipud(np.rot90(np.flipud(psf), 3))
                    elif iField == metr.nField+3:
                        psf = np.flipud(np.rot90(np.flipud(psf), 1))

                # below, we have 0 b/c we may have many
                stampFile = '%s/iter%d/sim%d_iter%d_wfs%d_%s_0.fits' % (
                    state.imageDir, state.iIter, state.iSim, state.iIter, iField,
                    self.wfsName[ioffset])
                if os.path.isfile(stampFile):
                    os.remove(stampFile)
                hdu = fits.PrimaryHDU(psf)
                hdu.writeto(stampFile)

                if debugLevel >= 3:
                    print('px = %d, py = %d' % (px, py))
                    print('offsetx = %d, offsety = %d' % (offsetx, offsety))
                    print('passed %d, %s' % (iField, self.wfsName[ioffset]))

        # make an image of the 8 donuts
        for iField in range(metr.nFieldp4-self.nWFS, metr.nFieldp4):
            chipStr, px, py = state.fieldXY2Chip(
                metr.fieldXp[iField], metr.fieldYp[iField], debugLevel)
            for ioffset in [0, 1]:
                src = glob.glob('%s/iter%d/sim%d_iter%d_wfs%d_%s_*.fits' % (
                    state.imageDir, state.iIter, state.iSim, state.iIter, iField,
                    self.wfsName[ioffset]))
                IHDU = fits.open(src[0])
                psf = IHDU[0].data
                IHDU.close()
                if state.inst[:4] == 'lsst':
                    nRow = 2
                    nCol = 4
                    if iField == metr.nField:
                        pIdx = 3 + ioffset  # 3 and 4
                    elif iField == metr.nField + 1:
                        pIdx = 1 + ioffset  # 1 and 2
                    elif iField == metr.nField + 2:
                        pIdx = 5 + ioffset  # 5 and 6
                    elif iField == metr.nField + 3:
                        pIdx = 7 + ioffset  # 7 and 8
                elif state.inst[:6] == 'comcam':
                    nRow = 3
                    nCol = 6
                    ic = np.floor(iField/nRow)
                    ir = iField%nRow
                    pIdx = int((nRow-ir-1)*nCol+ ic*2 +1 + ioffset) #does iField=0 give 13 and 14?
                    # print('pIdx = %d, chipStr= %s'%(pIdx, chipStr))
                plt.subplot(nRow, nCol, pIdx)
                plt.imshow(psf, origin='lower', interpolation='none')
                plt.title('%s_%s' % (chipStr, self.wfsName[ioffset]), fontsize=10)
                plt.axis('off')

        # plt.show()
        pngFile = '%s/iter%d/sim%d_iter%d_wfs.png' % (
            state.imageDir, state.iIter, state.iSim, state.iIter)
        plt.savefig(pngFile, bbox_inches='tight')

        #write out catalog for good wfs stars
        fid = open(self.catFile, 'w')
        for i in range(metr.nFieldp4-self.nWFS, metr.nFieldp4):
            intraFile = glob.glob('%s/iter%d/sim%d_iter%d_wfs%d_%s_*.fits' % (
                state.imageDir, state.iIter, state.iSim, state.iIter, i,
                self.wfsName[0]))[0]
            extraFile = glob.glob('%s/iter%d/sim%d_iter%d_wfs%d_%s_*.fits' % (
                state.imageDir, state.iIter, state.iSim, state.iIter, i,
                self.wfsName[1]))[0]
            if state.inst[:4] == 'lsst':
                if i == 31: 
                    fid.write('%9.6f %9.6f %9.6f %9.6f %s %s\n'% (
                        metr.fieldXp[i] - 0.020, metr.fieldYp[i],
                        metr.fieldXp[i] + 0.020, metr.fieldYp[i],
                        intraFile, extraFile))
                elif i == 32: 
                    fid.write('%9.6f %9.6f %9.6f %9.6f %s %s\n'% (
                        metr.fieldXp[i], metr.fieldYp[i] - 0.020,
                        metr.fieldXp[i], metr.fieldYp[i] + 0.020,
                        intraFile, extraFile))
                elif i == 33: 
                    fid.write('%9.6f %9.6f %9.6f %9.6f %s %s\n'% (
                        metr.fieldXp[i] + 0.020, metr.fieldYp[i],
                        metr.fieldXp[i] - 0.020, metr.fieldYp[i],
                        intraFile, extraFile))
                elif i == 34: 
                    fid.write('%9.6f %9.6f %9.6f %9.6f %s %s\n'% (
                        metr.fieldXp[i], metr.fieldYp[i] + 0.020,
                        metr.fieldXp[i], metr.fieldYp[i] - 0.020,
                        intraFile, extraFile))
            elif state.inst[:6] == 'comcam':
                fid.write('%9.6f %9.6f %9.6f %9.6f %s %s\n'% (
                    metr.fieldXp[i], metr.fieldYp[i],
                    metr.fieldXp[i], metr.fieldYp[i],
                        intraFile, extraFile))
        fid.close()
        
    def parallelCwfs(self, cwfsModel, numproc, debugLevel):
        fid = open(self.catFile)
        argList = []
        for line in fid:
            data = line.split()
            I1Field = [float(data[0]), float(data[1])]
            I2Field = [float(data[2]), float(data[3])]
            I1File = data[4]
            I2File = data[5]
            argList.append((I1File, I1Field, I2File, I2Field, self.inst, self.algo, cwfsModel))
            
            # test, pdb cannot go into the subprocess
            # aa = runcwfs(argList[0])
        #aa = runcwfs(argList[4])
                      
        pool = multiprocessing.Pool(numproc)
        zcarray = pool.map(runcwfs, argList)
        pool.close()
        pool.join()
        zcarray = np.array(zcarray)

        np.savetxt(self.zFile, zcarray)

    def checkZ4C(self, state, metr, debugLevel):
        z4c = np.loadtxt(self.zFile) #in micron
        z4cTrue = np.loadtxt(state.zTrueFile)
        
        x = range(4, self.znwcs+1)
        plt.figure(figsize=(10, 8))
        if state.inst[:4] == 'lsst':
            # subplots go like this
            #  2 1
            #  3 4
            pIdx = [2, 1, 3, 4]
            nRow = 2
            nCol = 2
        elif state.inst[:6] == 'comcam':
            pIdx = [7, 4, 1, 8, 5, 2, 9, 6, 3]
            nRow = 3
            nCol = 3
            
        for i in range(self.nWFS):
            chipStr, px, py = state.fieldXY2Chip(
                metr.fieldXp[i+metr.nFieldp4-self.nWFS], metr.fieldYp[i+metr.nFieldp4-self.nWFS], debugLevel)
            plt.subplot(nRow, nCol,pIdx[i])
            plt.plot(x, z4c[i,:self.znwcs3], label='CWFS',
             marker='o', color='r', markersize=6)
            plt.plot(x, z4cTrue[i+metr.nFieldp4-self.nWFS,3:self.znwcs], label='Truth',
             marker='.', color='b', markersize=10)
            if ( (state.inst[:4] == 'lsst' and (i==1 or i==2)) or
                (state.inst[:6] == 'comcam' and (i<=2)) ):
                plt.ylabel('$\mu$m')
            if ( (state.inst[:4] == 'lsst' and (i==2 or i==3)) or
                 (state.inst[:6] == 'comcam' and (i%nRow==0)) ):
                plt.xlabel('Zernike Index')
            leg = plt.legend(loc="best")
            leg.get_frame().set_alpha(0.5)        
            plt.grid()
            plt.title('Zernikes %s'%chipStr, fontsize=10)

        plt.savefig(self.zCompFile, bbox_inches='tight')
        
    def getZ4CfromBase(self, baserun, state):
        if not os.path.isfile(self.zFile):        
            baseFile = self.zFile.replace('sim%d'%state.iSim, 'sim%d'%baserun)
            os.link(baseFile, self.zFile)
        if not os.path.isfile(self.zCompFile):
            baseFile = self.zCompFile.replace('sim%d'%state.iSim, 'sim%d'%baserun)
            os.link(baseFile, self.zCompFile)
                        
def runcwfs(argList):
    I1File = argList[0]
    I1Field = argList[1]
    I2File = argList[2]
    I2Field = argList[3]
    inst = argList[4]
    algo = argList[5]
    model = argList[6]
    
    I1 = cwfsImage(I1File, I1Field, 'intra')
    I2 = cwfsImage(I2File, I2Field, 'extra')
    algo.reset(I1, I2)
    algo.runIt(inst, I1, I2, model)
    
    return np.append(algo.zer4UpNm*1e-3, algo.caustic)

