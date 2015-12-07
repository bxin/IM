#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import os
import glob
import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib.pyplot as plt

from cwfsAlgo import cwfsAlgo
from cwfsInstru import cwfsInstru


class aosWFS(object):

    def __init__(self, cwfsDir, instruFile, algoFile,
                 imgSizeinPix, wavelength, debugLevel):

        aosDir = os.getcwd()
        self.cwfsDir = cwfsDir
        os.chdir(cwfsDir)
        self.inst = cwfsInstru(instruFile, imgSizeinPix)
        self.algo = cwfsAlgo(algoFile, self.inst, debugLevel)
        os.chdir(aosDir)
        self.znwcs = self.algo.numTerms
        self.znwcs3 = self.znwcs - 3
        self.myZn = np.zeros((self.znwcs3 * 4, 2))
        self.phosimZn = np.zeros((self.znwcs3 * 4, 2))
        intrinsic35 = np.loadtxt('data/intrinsic_zn.txt')
        intrinsic35 = intrinsic35 * wavelength
        self.intrinsic4c = intrinsic35[
            -4:, 3:self.algo.numTerms].reshape((-1, 1))

        if debugLevel >= 3:
            print('znwcs3=%d' % self.znwcs3)
            print(self.intrinsic4c.shape)
            print(self.intrinsic4c[:5])

    def preprocess(self, state, metr, debugLevel):
        for iField in range(metr.nField, metr.nFieldp4):
            chipStr, px, py = state.fieldXY2Chip(
                metr.fieldXp[iField], metr.fieldYp[iField], debugLevel)
            src = glob.glob('%s/*%d*%s*' %
                            (state.imageDir, 90000 + state.iSim, chipStr))
            for ioffset in [0, 1]:
                chipFile = src[ioffset]
                IHDU = fits.open(chipFile)
                chipImage = IHDU[0].data
                IHDU.close()
                if ioffset == 1:
                    px = px - chipImage.shape[1]
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

                # read of corner raft are identical,
                # cwfs knows how to handle rotated images
                # flipud is b/c rot90 rotates images with origin at upper left
                if iField > metr.nField:
                    psf = np.flipud(
                        np.rot90(np.flipud(psf), iField - metr.nField))

                # below, we have 0 b/c we may have many
                stampFile = '%s/sim%d_iter%d_wfs%d_%s_0.fits' % (
                    state.imageDir, state.iSim, state.iIter, iField,
                    state.wfsName[ioffset])
                if os.path.isfile(stampFile):
                    os.remove(stampFile)
                hdu = fits.PrimaryHDU(psf)
                hdu.writeto(stampFile)

                if debugLevel >= 3:
                    print('px = %d, py = %d' % (px, py))
                    print('offsetx = %d, offsety = %d' % (offsetx, offsety))
                    print('passed %d, %s' % (iField, state.wfsName[ioffset]))

        for iField in range(metr.nField, metr.nFieldp4):
            chipStr, px, py = state.fieldXY2Chip(
                metr.fieldXp[iField], metr.fieldYp[iField], debugLevel)
            for ioffset in [0, 1]:
                src = glob.glob('%s/sim%d_iter%d_wfs%d_%s_*.fits' % (
                    state.imageDir, state.iSim, state.iIter, iField,
                    state.wfsName[ioffset]))
                IHDU = fits.open(src[0])
                psf = IHDU[0].data
                IHDU.close()
                if iField == metr.nField:
                    pIdx = 3 + ioffset  # 3 and 4
                elif iField == metr.nField + 1:
                    pIdx = 1 + ioffset  # 1 and 2
                elif iField == metr.nField + 2:
                    pIdx = 5 + ioffset  # 5 and 6
                elif iField == metr.nField + 3:
                    pIdx = 7 + ioffset  # 7 and 8

                plt.subplot(2, 4, pIdx)
                plt.imshow(psf, origin='lower')
                plt.title('%s_%s' % (chipStr, state.wfsName[ioffset]))
                plt.axis('off')

        # plt.show()
        pngFile = '%s/sim%d_iter%d_wfs.png' % (
            state.imageDir, state.iSim, state.iIter)
        plt.savefig(pngFile, bbox_inches='tight')
