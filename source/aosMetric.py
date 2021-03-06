#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import os
import sys
import multiprocessing

import numpy as np
import scipy.special as sp
from astropy.io import fits

from lsst.cwfs.tools import padArray
from lsst.cwfs.tools import extractArray
from lsst.cwfs.tools import ZernikeAnnularFit
from lsst.cwfs.tools import ZernikeAnnularEval

from lsst.cwfs.errors import nonSquareImageError
from aosErrors import psfSamplingTooLowError
from aosTeleState import aosTeleState

import matplotlib.pyplot as plt


class aosMetric(object):

    def __init__(self, instName, opdSize, znwcs3, debugLevel, pixelum=10):
        aosSrcDir = os.path.split(os.path.abspath(__file__))[0]            
        if instName[:4] == 'lsst':
            self.nArm = 6
            armLen = [0.379, 0.841, 1.237, 1.535, 1.708]
            armW = [0.2369, 0.4786, 0.5689, 0.4786, 0.2369]
            self.nRing = len(armLen)
            self.nField = self.nArm * self.nRing + 1
            self.nFieldp4 = self.nField + 4
            self.fieldX = np.zeros(self.nFieldp4)
            self.fieldY = np.zeros(self.nFieldp4)
            self.fieldX[0] = 0
            self.fieldY[0] = 0
            self.w = [0]
            for i in range(self.nRing):
                self.w = np.concatenate((self.w, np.ones(self.nArm) * armW[i]))
                self.fieldX[i * self.nArm + 1: (i + 1) * self.nArm + 1] =\
                    armLen[i] * np.cos(np.arange(self.nArm) *
                                       (2 * np.pi) / self.nArm)
                self.fieldY[i * self.nArm + 1: (i + 1) * self.nArm + 1] =\
                    armLen[i] * np.sin(np.arange(self.nArm) *
                                       (2 * np.pi) / self.nArm)
            # self.fieldX[self.nField:]=[1.185, -1.185, -1.185, 1.185]
            # self.fieldY[self.nField:]=[1.185, 1.185, -1.185, -1.185]
            # counter-clock wise
            self.fieldX[self.nField:] = [1.176, -1.176, -1.176, 1.176]
            self.fieldY[self.nField:] = [1.176, 1.176, -1.176, -1.176]

            self.fwhmModelFileBase = '%s/../data/fwhmModel/fwhm_vs_z_500nm' % aosSrcDir

        elif instName[:6] == 'comcam':
            nRow = 3
            nCol = 3
            self.nField = nRow * nCol
            self.fieldX = np.zeros(self.nField)
            self.fieldY = np.zeros(self.nField)
            sensorD = 0.2347

            for i in range(nRow):
                for j in range(nCol):
                    self.fieldX[i * nRow + j] = (i - 1) * sensorD
                    self.fieldY[i * nRow + j] = (j - 1) * sensorD

            self.w = np.ones(self.nField)
            self.nFieldp4 = self.nField

        self.w = self.w / np.sum(self.w)

        # below, p is for PSF
        self.fieldXp = self.fieldX.copy()
        self.fieldYp = self.fieldY.copy()

        if instName[:4] == 'lsst' and pixelum == 10:  # falling on chip edge
            self.fieldXp[19] += 0.004
            self.fieldXp[22] -= 0.004

        if debugLevel >= 3:
            print(self.w.shape)
            print(self.w)

        aa = np.loadtxt('%s/../data/pssn_alpha.txt'%aosSrcDir)
        self.pssnAlpha = aa[:, 0]
        # self.pssnRange = aa[: 1]

        self.znx2 = np.zeros((self.nFieldp4, znwcs3))
        self.stampD = 2**np.ceil(np.log2(opdSize))

    def getFWHMfromZ(self):
        self.fwhm = np.zeros(self.nField)

    def getPSSNfromZ(self):
        pass

    def getFFTPSF(self, fftpsfoff, state, imagedelta, numproc,
                  debugLevel, sensorfactor=1, fno=1.2335):

        if not fftpsfoff:
            argList = []
            for i in range(self.nField):
                opdFile = '%s/iter%d/sim%d_iter%d_opd%d.fits' % (
                    state.imageDir, state.iIter, state.iSim, state.iIter, i)
                psfFile = opdFile.replace('opd', 'fftpsf')
                argList.append((opdFile, state, imagedelta,
                                sensorfactor, fno, psfFile,
                                debugLevel))
                if sys.platform == 'darwin':
                    runFFTPSF(argList[i])

            if sys.platform != 'darwin':
                # test, pdb cannot go into the subprocess
                # runFFTPSF(argList[0])
                pool = multiprocessing.Pool(numproc)
                pool.map(runFFTPSF, argList)
                pool.close()
                pool.join()

            plt.figure(figsize=(10, 10))
            for i in range(self.nField):

                psfFile = '%s/iter%d/sim%d_iter%d_fftpsf%d.fits' % (
                    state.imageDir, state.iIter, state.iSim, state.iIter, i)
                IHDU = fits.open(psfFile)
                psf = IHDU[0].data
                IHDU.close()

                if state.inst[:4] == 'lsst':
                    if i == 0:
                        pIdx = 1
                    else:
                        pIdx = i + self.nArm
                    nRow = self.nRing + 1
                    nCol = self.nArm
                elif state.inst[:6] == 'comcam':
                    aa = [7, 4, 1, 8, 5, 2, 9, 6, 3]
                    pIdx = aa[i]
                    nRow = 3
                    nCol = 3
                displaySize = 100
                plt.subplot(nRow, nCol, pIdx)
                plt.imshow(extractArray(psf, displaySize),
                           origin='lower', interpolation='none')
                plt.title('%d' % i)
                plt.axis('off')

            # plt.show()
            pngFile = '%s/iter%d/sim%d_iter%d_fftpsf.png' % (
                state.imageDir, state.iIter, state.iSim, state.iIter)
            plt.savefig(pngFile, bbox_inches='tight')
            plt.close()

    def getPSSNandMore(self, pssnoff, state, numproc,
                       debugLevel,
                       outFile='', pixelum=0):
        """
        pixelum = 0: the input is opd map
        pixelum != 0: input is a fine-pixel PSF image stamp
        """
        if not outFile:
            outFile = self.PSSNFile

        if not pssnoff:
            # multithreading on MacOX doesn't work with pinv
            # before we calc_pssn, we do ZernikeFit to remove PTT
            # pinv appears in ZernikeFit()
            if sys.platform == 'darwin':
                self.PSSNw = np.zeros((self.nField, state.nOPDw))
            argList = []
            icount = 0
            for i in range(self.nField):
                for irun in range(state.nOPDw):
                    inputFile = []
                    if pixelum > 0:
                        inputFile.append(
                            '%s/iter%d/sim%d_iter%d_psf%d.fits' % (
                            state.imageDir, state.iIter, state.iSim,
                            state.iIter,
                            i))
                    elif pixelum < 0:
                        inputFile.append(
                            '%s/iter%d/sim%d_iter%d_fftpsf%d.fits' % (
                            state.imageDir, state.iIter, state.iSim,
                            state.iIter,
                            i))
                        
                    if state.nOPDw == 1:
                        inputFile.append(
                            '%s/iter%d/sim%d_iter%d_opd%d.fits' % (
                            state.imageDir, state.iIter, state.iSim,
                            state.iIter, i))
                        wlum = state.wavelength
                    else:
                        inputFile.append(
                            '%s/iter%d/sim%d_iter%d_opd%d_w%d.fits' % (
                            state.imageDir, state.iIter, state.iSim,
                            state.iIter, i, irun))
                        wlum = aosTeleState.GQwave[state.band][irun]
                    argList.append((inputFile, state,
                                        wlum, debugLevel, pixelum))

                    if sys.platform == 'darwin':
                        self.PSSNw[i, irun] = runPSSNandMore(argList[icount])
                    icount += 1
                    
            if sys.platform != 'darwin':
                # test, pdb cannot go into the subprocess
                # aa = runPSSNandMore(argList[0])
                pool = multiprocessing.Pool(numproc)
                self.PSSNw = pool.map(runPSSNandMore, argList)
                pool.close()
                pool.join()
                self.PSSNw = np.array(self.PSSNw).reshape(self.nField, -1)

            wt = np.tile(np.array(aosTeleState.GQwt[state.band]),
                             (self.nField,1))
            self.PSSN = np.sum(wt * self.PSSNw, axis=1)
            self.FWHMeff = 1.086 * 0.6 * np.sqrt(1 / self.PSSN - 1)
            self.dm5 = -1.25 * np.log10(self.PSSN)

            if debugLevel >= 2:
                for i in range(self.nField):
                    print('---field#%d, PSSN=%7.4f, FWHMeff = %5.0f mas' % (
                        i, self.PSSN[i], self.FWHMeff[i] * 1e3))

            self.GQPSSN = np.sum(self.w * self.PSSN)
            self.GQFWHMeff = np.sum(self.w * self.FWHMeff)
            self.GQdm5 = np.sum(self.w * self.dm5)
            a1 = np.concatenate((self.PSSN, self.GQPSSN * np.ones(1)))
            a2 = np.concatenate((self.FWHMeff, self.GQFWHMeff * np.ones(1)))
            a3 = np.concatenate((self.dm5, self.GQdm5 * np.ones(1)))
            np.savetxt(outFile, np.vstack((a1, a2, a3)))

            if debugLevel >= 2:
                print(self.GQPSSN)
        else:
            aa = np.loadtxt(outFile)
            self.GQFWHMeff = aa[1, -1]  # needed for shiftGear

    def getPSSNandMorefromBase(self, baserun, state):
        if not os.path.isfile(self.PSSNFile):
            baseFile = self.PSSNFile.replace(
                'sim%d' % state.iSim, 'sim%d' % baserun)
            os.link(baseFile, self.PSSNFile)
        aa = np.loadtxt(self.PSSNFile)
        self.GQFWHMeff = aa[1, -1]  # needed for shiftGear

    def getEllipticity(self, ellioff, state, numproc,
                       debugLevel,
                       outFile='', pixelum=0):
        """
        pixelum = 0: the input is opd map
        pixelum != 0: input is a fine-pixel PSF image stamp
        """
        if not outFile:
            outFile = self.elliFile

        if not ellioff:
            # multithreading on MacOX doesn't work with pinv
            # before we psf2eAtmW(), we do ZernikeFit to remove PTT
            # pinv appears in ZernikeFit()
            if sys.platform == 'darwin':
                self.elliw = np.zeros((self.nField, state.nOPDw))
            argList = []
            icount = 0
            for i in range(self.nField):
                for irun in range(state.nOPDw):
                    inputFile = []
                    if pixelum > 0:
                        inputFile.append(
                            '%s/iter%d/sim%d_iter%d_psf%d.fits' % (
                            state.imageDir, state.iIter, state.iSim,
                            state.iIter,
                            i))
                    elif pixelum < 0:
                        inputFile.append(
                            '%s/iter%d/sim%d_iter%d_fftpsf%d.fits' % (
                            state.imageDir, state.iIter, state.iSim,
                            state.iIter,
                            i))

                    if state.nOPDw == 1:
                        inputFile.append(
                            '%s/iter%d/sim%d_iter%d_opd%d.fits' % (
                            state.imageDir, state.iIter, state.iSim,
                            state.iIter, i))
                        wlum = state.wavelength
                    else:
                        inputFile.append(
                            '%s/iter%d/sim%d_iter%d_opd%d_w%d.fits' % (
                            state.imageDir, state.iIter, state.iSim,
                            state.iIter, i, irun))
                        wlum = aosTeleState.GQwave[state.band][irun]
                    argList.append((inputFile, state,
                                    wlum, debugLevel, pixelum))

                    if sys.platform == 'darwin':
                        self.elliw[i, irun] = runEllipticity(argList[icount])
                    icount += 1

            if sys.platform != 'darwin':
                pool = multiprocessing.Pool(numproc)
                self.elliw = pool.map(runEllipticity, argList)
                pool.close()
                pool.join()
                self.elliw = np.array(self.elliw).reshape(self.nField, -1)

            wt = np.tile(np.array(aosTeleState.GQwt[state.band]),
                             (self.nField,1))
            self.elli = np.sum(wt * self.elliw, axis = 1)
            for i in range(self.nField):
                if debugLevel >= 2:
                    print('---field#%d, elli=%7.4f' % (i, self.elli[i]))

            self.GQelli = np.sum(self.w * self.elli)
            a1 = np.concatenate((self.elli, self.GQelli * np.ones(1)))
            np.savetxt(outFile, a1)
            if debugLevel >= 2:
                print(self.GQelli)

    def getEllipticityfromBase(self, baserun, state):
        if not os.path.isfile(self.elliFile):
            baseFile = self.elliFile.replace(
                'sim%d' % state.iSim, 'sim%d' % baserun)
            os.link(baseFile, self.elliFile)


def calc_pssn(array, wlum, type='opd', D=8.36, r0inmRef=0.1382, zen=0,
              pmask=0, imagedelta=0, fno=1.2335, debugLevel=0):
    """
    array: the array that contains either opd or pdf
           opd need to be in microns
    wlum: wavelength in microns
    type: what is used to calculate pssn - either opd or psf
    psf doesn't matter, will be normalized anyway
    D: side length of OPD image in meter
    r0inmRef: fidicial atmosphere r0@500nm in meter, Konstantinos uses 0.20
    Now that we use vonK atmosphere, r0in=0.1382 -> fwhm=0.6"
    earlier, we used Kolm atmosphere, r0in=0.1679 -> fwhm=0.6"
    zen: telescope zenith angle

    The following are only needed when the input array is psf -
    pmask: pupil mask. when opd is used, it can be generated using opd image,
    we can put 0 or -1 or whatever here.
    when psf is used, this needs to be provided separately with same
    size as array.
    imagedelta and fno are only needed when psf is used. use 0,0 for opd

    THE INTERNAL RESOLUTION THAT FFTS OPERATE ON IS VERY IMPORTANT
    TO THE ACCUARCY OF PSSN.
    WHEN TYPE='OPD', NRESO=SIZE(ARRAY,1)
    WHEN TYPE='PSF', NRESO=SIZE(PMASK,1)
       for the psf option, we can not first convert psf back to opd then
       start over,
       because psf=|exp(-2*OPD)|^2. information has been lost in the | |^2.
       we need to go forward with psf->mtf,
       and take care of the coordinates properly.
    """

    if array.ndim == 3:
        array2D = array[0, :, :].squeeze()

    if type == 'opd':
        try:
            m = max(array2D.shape)
        except NameError:
            m = max(array.shape)
        k = 1
        imagedelta = fno * wlum
    else:
        m = max(pmask.shape)
        # pupil needs to be padded k times larger to get imagedelta
        k = fno * wlum / imagedelta

    mtfa = createMTFatm(D, m, k, wlum, zen, r0inmRef)

    if type == 'opd':
        try:
            iad = (array2D != 0)
        except NameError:
            iad = (array != 0)
    elif type == 'psf':
        mk = int(m + np.rint((m * (k - 1) + 1e-5) / 2) * 2)  # add even number
        iad = pmask  # padArray(pmask, m)

    # number of non-zero elements, used for normalization later
    # miad2 = np.count_nonzero(iad)

    # Perfect telescope
    opdt = np.zeros((m, m))
    psft = opd2psf(opdt, iad, wlum, imagedelta, 1, fno, debugLevel)
    otft = psf2otf(psft)  # OTF of perfect telescope
    otfa = otft * mtfa  # add atmosphere to perfect telescope
    psfa = otf2psf(otfa)
    pssa = np.sum(psfa**2)  # atmospheric PSS = 1/neff_atm

    # Error;
    if type == 'opd':
        if array.ndim == 2:
            ninst = 1
        else:
            ninst = array.shape[0]
        for i in range(ninst):
            if array.ndim == 2:
                array2D = array
            else:
                array2D = array[i, :, :].squeeze()
            psfei = opd2psf(array2D, iad, wlum, 0, 0, 0, debugLevel)
            if i == 0:
                psfe = psfei
            else:
                psfe += psfei
        psfe = psfe / ninst
    else:
        if array.shape[0] == mk:
            psfe = array
        elif array.shape[0] > mk:
            psfe = extractArray(array, mk)
        else:
            print('calc_pssn: image provided too small, %d < %d x %6.4f' % (
                array.shape[0], m, k))
            print('IQ is over-estimated !!!')
            psfe = padArray(array, mk)

        psfe = psfe / np.sum(psfe) * np.sum(psft)

    pixmas = imagedelta * 20
    aa = psfe/np.sum(psfe)
    neff = 1/np.sum(aa**2)
    fwhmeff = 0.664*pixmas * np.sqrt(neff)
    
    otfe = psf2otf(psfe)  # OTF of error
    otftot = otfe * mtfa  # add atmosphere to error
    psftot = otf2psf(otftot)
    pss = np.sum(psftot**2)  # atmospheric + error PSS

    pssn = pss / pssa  # normalized PSS
    if debugLevel >= 3:
        print('pssn = %10.8e/%10.8e = %6.4f' % (pss, pssa, pssn))

    return pssn, fwhmeff


def createMTFatm(D, m, k, wlum, zen, r0inmRef):
    """
    m is the number of pixel we want to have to cover the length of D.
    If we want a k-times bigger array, we pad the mtf generated using k=1.
    """

    sfa = atmSF('vonK', D, m, wlum, zen, r0inmRef)
    mtfa = np.exp(-0.5 * sfa)

    N = int(m + np.rint((m * (k - 1) + 1e-5) / 2) * 2)  # add even number
    mtfa = padArray(mtfa, N)

    return mtfa


def atmSF(model, D, m, wlum, zen, r0inmRef):
    """
    create the atmosphere phase structure function
    model = 'Kolm'
             = 'vonK'
    """
    r0a = r0Wz(r0inmRef, zen, wlum)
    L0 = 30  # outer scale in meter, only used when model=vonK

    m0 = np.rint(0.5 * (m + 1) + 1e-5)
    aa = np.arange(1, m + 1)
    x, y = np.meshgrid(aa, aa)

    dr = D / (m - 1)  # frequency resolution in 1/rad
    r = dr * np.sqrt((x - m0)**2 + (y - m0)**2)

    if model == 'Kolm':
        sfa = 6.88 * (r / r0a)**(5 / 3)
    elif model == 'vonK':
        sfa_c = 2 * sp.gamma(11 / 6) / 2**(5 / 6) / np.pi**(8 / 3) *\
            (24 / 5 * sp.gamma(6 / 5))**(5 / 6) * (r0a / L0)**(-5 / 3)
        # modified bessel of 2nd/3rd kind
        sfa_k = sp.kv(5 / 6, (2 * np.pi / L0 * r))
        sfa = sfa_c * (2**(-1 / 6) * sp.gamma(5 / 6) -
                       (2 * np.pi / L0 * r)**(5 / 6) * sfa_k)

        # if we don't do below, everything will be nan after ifft2
        # midp = r.shape[0]/2+1
        # 1e-2 is to avoid x.49999 be rounded to x
        midp = int(np.rint(0.5 * (r.shape[0] - 1) + 1e-2))
        sfa[midp, midp] = 0  # at this single point, sfa_k=Inf, 0*Inf=Nan;

    return sfa


def r0Wz(r0inmRef, zen, wlum):
    zen = zen * np.pi / 180.  # telescope zenith angle, change here
    r0aref = r0inmRef * np.cos(zen)**0.6  # atmosphere reference r0
    r0a = r0aref * (wlum / 0.5)**1.2  # atmosphere r0, a function of wavelength
    return r0a


def psf2eAtmW(array, wlum, type='opd', D=8.36, pmask=0, r0inmRef=0.1382,
              sensorFactor=1,
              zen=0, imagedelta=0.2, fno=1.2335, debugLevel=0):
    """
    array: wavefront OPD in micron, or psf image
    unlike calc_pssn(), here imagedelta needs to be provided for type='opd'
        because the ellipticity calculation operates on psf.

    """
    k = fno * wlum / imagedelta
    if type == 'opd':
        m = array.shape[0] / sensorFactor
        psfe = opd2psf(array, 0, wlum, imagedelta,
                       sensorFactor, fno, debugLevel)
    else:
        m = max(pmask.shape)
        psfe = array
    mtfa = createMTFatm(D, m, k, wlum, zen, r0inmRef)

    otfe = psf2otf(psfe)  # OTF of error

    otf = otfe * mtfa
    psf = otf2psf(otf)

    if debugLevel >= 3:
        print('Below from the Gaussian weigting function on elli')
    e, q11, q22, q12 = psf2eW(psf, imagedelta, wlum, 'Gau', debugLevel)

    return e, q11, q22, q12


def psf2eW(psf, pixinum, wlum, atmModel, debugLevel=0):

    x, y = np.meshgrid(np.arange(1, psf.shape[0] + 1),
                       np.arange(1, psf.shape[1] + 1))
    xbar = np.sum(x * psf) / np.sum(psf)
    ybar = np.sum(y * psf) / np.sum(psf)

    r2 = (x - xbar)**2 + (y - ybar)**2

    fwhminarcsec = 0.6
    oversample = 1
    W = createAtm(atmModel, wlum, fwhminarcsec, r2, pixinum, oversample,
                  0, '', debugLevel)

    if debugLevel >= 3:
        print('xbar=%6.3f, ybar=%6.3f' % (xbar, ybar))
        # plot.plotImage(psf,'')
        # plot.plotImage(W,'')

    psf = psf * W  # apply weighting function

    Q11 = np.sum(((x - xbar)**2) * psf) / np.sum(psf)
    Q22 = np.sum(((y - ybar)**2) * psf) / np.sum(psf)
    Q12 = np.sum(((x - xbar) * (y - ybar)) * psf) / np.sum(psf)

    T = Q11 + Q22
    if T > 1e-20:
        e1 = (Q11 - Q22) / T
        e2 = 2 * Q12 / T

        e = np.sqrt(e1**2 + e2**2)
    else:
        e = 0

    return e, Q11, Q22, Q12


def createAtm(model, wlum, fwhminarcsec, gridsize, pixinum, oversample,
              cutOutput, outfile, debugLevel):
    """
    gridsize can be int or an array. When it is array, it is r2
    cutOutput only applies to Kolm and vonK
    """
    if isinstance(gridsize, (int)):
        nreso = gridsize * oversample
        nr = nreso / 2  # n for radius length
        aa = np.linspace(-nr + 0.5, nr - 0.5, nreso)
        x, y = np.meshgrid(aa)
        r2 = x * x + y * y
    else:
        r2 = gridsize

    if model[:4] == 'Kolm' or model[:4] == 'vonK':
        pass
    else:
        fwhminum = fwhminarcsec / 0.2 * 10
        if model == 'Gau':
            sig = fwhminum / 2 / np.sqrt(2 * np.log(2))  # in micron
            sig = sig / (pixinum / oversample)
            z = np.exp(-r2 / 2 / sig**2)
        elif model == '2Gau':
            # below is used to manually solve for sigma
            # let x = exp(-r^2/(2*alpha^2)), which results in 1/2*max
            # we want to get (1+.1)/2=0.55 from below
            # x=0.4673194304;printf('%20.10f\n'%x**.25*.1+x);
            sig = fwhminum / (2 * np.sqrt(-2 * np.log(0.4673194304)))
            sig = sig / (pixinum / oversample)  # in (oversampled) pixel
            z = np.exp(-r2 / 2 / sig**2) + 0.4 / 4 * np.exp(-r2 / 8 / sig**2)
        if debugLevel >= 3:
            print('sigma1=%6.4f arcsec' %
                  (sig * (pixinum / oversample) / 10 * 0.2))
    return z


def opd2psf(opd, pupil, wavelength, imagedelta, sensorFactor, fno, debugLevel):
    """
    wavefront OPD in micron
    imagedelta in micron, use 0 if pixel size is not specified
    wavelength in micron

    if pupil is a number, not an array, we will get pupil geometry from opd
    The following are not needed if imagedelta=0,
    sensorFactor, fno
    """

    opd[np.isnan(opd)] = 0
    try:
        if (pupil.shape == opd.shape):
            pass
        else:
            raise AttributeError
    except AttributeError:
        pupil = (opd != 0)

    if imagedelta != 0:
        try:
            if opd.shape[0] != opd.shape[1]:
                raise(nonSquareImageError)
        except nonSquareImageError:
            print('Error (opd2psf): Only square images are accepted.')
            print('image size = (%d, %d)' % (
                opd.shape[0], opd.shape[1]))
            sys.exit()

        k = fno * wavelength / imagedelta
        padding = k / sensorFactor
        try:
            if padding < 1:
                raise(psfSamplingTooLowError)
        except psfSamplingTooLowError:
            print('opd2psf: sampling too low, data inaccurate')
            print('imagedelta needs to be smaller than fno*wlum=%4.2f um' % (
                fno * wavelength))
            print('         so that the padding factor > 1')
            print('         otherwise we have to cut pupil to be < D')
            sys.exit()

        sensorSamples = opd.shape[0]
        # add even number for padding
        N = int(sensorSamples + \
            np.rint(((padding - 1) * sensorSamples + 1e-5) / 2) * 2)
        pupil = padArray(pupil, N)
        opd = padArray(opd, N)
        if debugLevel >= 3:
            print('padding=%8.6f' % padding)
    # if imagedelta = 0, we don't do any padding, and go with below
    z = pupil * np.exp(-2j * np.pi * opd / wavelength)
    z = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(z),
                                    s=z.shape))  # /sqrt(miad2/m^2)
    z = np.absolute(z**2)
    z = z / np.sum(z)

    if debugLevel >= 3:
        print('opd2psf(): imagedelta=%8.6f' % imagedelta, end='')
        if imagedelta == 0:
            print('0 means using OPD with padding as provided')
        else:
            print('')
        print('verify psf has been normalized: %4.1f' % np.sum(z))

    return z


def psf2otf(psf):
    otf = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf),
                                      s=psf.shape))
    return otf


def otf2psf(otf):
    psf = np.absolute(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(otf),
                                                   s=otf.shape)))
    return psf


def runEllipticity(argList):
    inputFile = argList[0]
    opdx = argList[1].opdx
    opdy = argList[1].opdy
    wavelength = argList[2]
    debugLevel = argList[3]
    pixelum = np.abs(argList[4])
    print('runEllipticity: %s ' % inputFile)

    if pixelum == 0:
        IHDU = fits.open(inputFile[0])
        opd = IHDU[0].data  # um
        IHDU.close()

        # before psf2eAtmW()
        # (1) remove PTT,
        # (2) make sure outside of pupil are all zeros
        idx = (opd != 0)
        Z = ZernikeAnnularFit(opd[idx], opdx[idx], opdy[idx], 3, 0)
        Z[3:] = 0
        opd[idx] -= ZernikeAnnularEval(Z, opdx[idx], opdy[idx], 0)

        elli, _, _, _ = psf2eAtmW(
            opd, wavelength, debugLevel=debugLevel)
    else:
        IHDU = fits.open(inputFile[0])
        psf = IHDU[0].data  # unit: um
        IHDU.close()

        # opd only needed to help determine how big mtfa needs to be
        IHDU = fits.open(inputFile[1])
        opd = IHDU[0].data  # unit: um
        IHDU.close()
        iad = (opd != 0)

        elli, _, _, _ = psf2eAtmW(
            psf, wavelength, type='psf', pmask=iad,
            imagedelta=pixelum, debugLevel=debugLevel)

    return elli


def runPSSNandMore(argList):
    """
    pixelum = 0 means we use opd, meanwhile only opd is provided.
    pixelum !=0 means we use psf. both psf and pmask needs to be provided.
    """

    inputFile = argList[0]
    opdx = argList[1].opdx
    opdy = argList[1].opdy
    wavelength = argList[2]
    debugLevel = argList[3]
    pixelum = np.abs(argList[4])
    print('runPSSNandMore: %s ' % inputFile)

    if pixelum == 0:
        IHDU = fits.open(inputFile[0])
        opd = IHDU[0].data  # unit: um
        IHDU.close()

        # before calc_pssn,
        # (1) remove PTT,
        # (2) make sure outside of pupil are all zeros
        idx = (opd != 0)
        Z = ZernikeAnnularFit(opd[idx], opdx[idx], opdy[idx], 3, 0)
        Z[3:] = 0
        opd[idx] -= ZernikeAnnularEval(Z, opdx[idx], opdy[idx], 0)

        pssn, fwhmeff = calc_pssn(opd, wavelength, debugLevel=debugLevel)
    else:
        IHDU = fits.open(inputFile[0])
        psf = IHDU[0].data  # unit: um
        IHDU.close()

        # opd only needed to help determine pupil geometry
        IHDU = fits.open(inputFile[1])
        opd = IHDU[0].data  # unit: um
        IHDU.close()
        iad = (opd != 0)

        pssn, fwhmeff = calc_pssn(psf, wavelength, type='psf', pmask=iad,
                         imagedelta=pixelum,
                         debugLevel=debugLevel)

    return pssn


def runFFTPSF(argList):
    opdFile = argList[0]
    opdx = argList[1].opdx
    opdy = argList[1].opdy
    wavelength = argList[1].effwave
    imagedelta = argList[2]
    sensorfactor = argList[3]
    fno = argList[4]
    psfFile = argList[5]
    debugLevel = argList[6]
    print('runFFTPSF: %s ' % opdFile)

    IHDU = fits.open(opdFile)
    opd = IHDU[0].data  # unit: um
    IHDU.close()

    # before opd2psf,
    # (1) remove PTT, (for consistence with calc_pssn,
    #        in principle doesn't matter,
    # in practice, this affects centering, so it affects edge cutoff on psf)
    # (2) make sure outside of pupil are all zeros
    idx = (opd != 0)
    Z = ZernikeAnnularFit(opd[idx], opdx[idx], opdy[idx], 3, 0)
    Z[3:] = 0
    opd[idx] -= ZernikeAnnularEval(Z, opdx[idx], opdy[idx], 0)

    psf = opd2psf(opd, 0, wavelength, imagedelta, sensorfactor,
                  fno, debugLevel)
    if os.path.isfile(psfFile):
        os.remove(psfFile)
    hdu = fits.PrimaryHDU(psf)
    hdu.writeto(psfFile)

def psf2delta(psf, pixelSize, delta, cutoffI, metric, watm):
    '''
    % metric = 'fwhm'
    %           this routine calculates delta80, then convert to FWHM (same as psf2FWHM80())
    % metric = 'fwhm99'
    %           this routine calculates delta99, then convert to FWHM

    % metric = '' or anything else
    %           calculates deltaxx, as in xx% encircled energy.

    % when watm=0, the conversion (from delta80 to FWHM) is based on single Gaussian
    % when watm=1, the conversion (from delta80 to FWHM) is based on double Gaussian (as defined in SRD)
    %               (before we convolve with Double Gaussian first)

    %use the intensity centroid (xbar, ybar) as the center
    %get deltaxx (xx% encircled energy)
    % for example, xx=0.8, 0.95, 0.99
    %for FWHM, use metric='fwhm', 
    % what we do is to get delta80 (80% encircled energy)
    % then multiply by 2/sqrt(log(5)/log(2))=1.3125

    % when using metric='fwhm99'
    % what we do is to get delta99 (99% encircled energy)
    % then multiply by 2/sqrt(log(100)/log(2))=0.7759
    '''
    if watm:
        # atm2D=load('aosSim/atmGau.txt');%pixel size = 2um
        atm2D=load('aosSim/atm2Gau.txt') #pixel size = 2um
        atmPixelSize=2
        downf=atmPixelSize/pixelSize
        psfcut=mod(size(psf,1),downf)
        psf=extractArray(psf,size(psf,1)-psfcut)
        psf=downResolution(psf,downf,size(psf,1)/downf,size(psf,2)/downf)
        psf=conv2(atm2D,psf,'same')

    if metric == 'fwhm':
        delta=0.8
    elif metric == 'fwhm99':
        delta=0.99

    psf=psf/np.max(psf) #normalize peak intensity to 1
    m = psf.shape[0]
    n = psf.shape[1]
    x, y = np.meshgrid(np.arange(1, m + 1), np.arange(1, n + 1))

    sumI=np.sum(psf)
    xbar=np.sum(x*psf)/sumI
    ybar=np.sum(y*psf)/sumI

    psf = np.roll(psf, int(np.round(psf.shape[1]/2-ybar)), axis = 0)
    psf = np.roll(psf, int(np.round(psf.shape[0]/2-xbar)), axis = 1)
    xbarc=np.sum(x*psf)/sumI
    ybarc=np.sum(y*psf)/sumI

    r2=(x-xbarc)**2+(y-ybarc)**2
    mn=m*n
    aa = r2.reshape((mn,1))[:,0]
    idx=np.argsort(aa)
    r2sort = aa[idx]

    if cutoffI>0:
        maskR=np.sqrt(np.max(r2[psf>cutoffI]))
        psf[r2>maskR**2]=0

    psf[psf<cutoffI]=0 #Not needed when use maskR to mask off edges???
    sumI=np.sum(psf)

    sumIxx=delta*sumI

    mysum=0
    psf1d = psf.reshape((mn,1))[:,0]
    for i in range(len(idx)):
        mysum=mysum+psf1d[idx[i]]
        if mysum>sumIxx:
            break

    FWHM=np.sqrt(r2sort[i]) #this is actually the xx% encircled energy

    if watm:
        FWHM=FWHM*downf

    FWHM=FWHM*pixelSize
    if metric == 'fwhm':
        if watm:
            #for G1+0.4*G2
            # below is used to manually solve for delta80=a*sigma
            # below, x is exp(-r^2/(2*alpha^2)), which results in integration
            # from 0 to r to be 0.8
            # we want to get 1-0.8=0.2 from below
            #     x=0.072464801;fprintf('%20.10f\n',10/14*(x^.25*.4+x));
            # delta80=2.2911*sigma
            # FWHM=2.4670*sigma %  -->see createAtmTxt.m, 2*sqrt(-2*log(0.4673194304))=2.4670
            # 1.0768 below = 2.4670/2.2911
            FWHM=1.0768*FWHM #convert delta80 into FWHM, assuming it is double Gaussian
        
            #for delta90
            # x=0.0108632438;fprintf('%20.10f\n',10/14*(x^.25*.4+x))
            # FWHM=3.0074*sigma
            #for delta95
            # x=0.0008911;fprintf('%20.10f\n',10/14*(x^.25*.4+x))
            # FWHM=3.7478*sigma
            #for delta99
            # x=0.0000015;fprintf('%20.10f\n',10/14*(x^.25*.4+x))
            # FWHM=5.1788*sigma
        else:
            # for single Gaussian
            FWHM=1.3125*FWHM

    elif metric == 'fwhm99':
        # for single Gaussian
        FWHM=0.7759*FWHM

    return FWHM, xbar, ybar

def psf2FWHMrms(psf, maskR, cutoffI):
    ''' this routine calculates FWHM based on RMS (AndyR's note)
    %use maskR<0 for no mask
    %use cutoffI<0 for no cutoff on intensity
    %most commonly only one out of the two options above is used
'''
    m = psf.shape[0]
    n = psf.shape[1]
    x, y = np.meshgrid(np.arange(1, m + 1), np.arange(1, n + 1))

    sumI=np.sum(psf)
    xbar=np.sum(x*psf)/sumI
    ybar=np.sum(y*psf)/sumI

    psf = np.roll(psf, int(np.round(psf.shape[1]/2-ybar)), axis = 0)
    psf = np.roll(psf, int(np.round(psf.shape[0]/2-xbar)), axis = 1)
    xbarc=np.sum(x*psf)/sumI
    ybarc=np.sum(y*psf)/sumI

    r2=(x-xbarc)**2+(y-ybarc)**2    

    if maskR>0:
        psf[r2>maskR**2]=0

    if cutoffI>0:
        maskR=np.sqrt(np.max(r2[psf/np.max(psf)>cutoffI]))
        psf[r2>maskR**2]=0
        
    psfn = psf/np.sum(psf)
    FWHM=np.sqrt( np.sum(r2*psfn) )*1.665;

    return FWHM, xbar, ybar, maskR

def psf2FWHMring(array, wlum, type='opd', D=8.36, r0inmRef=0.1382, zen=0,
              pmask=0, imagedelta=0, fno=1.2335, fwhm_thresh=0.01,
                     power = 2, debugLevel=0):
    '''
    wavefront OPD in micron
'''
    wl = wlum*1.e-6
    if array.ndim == 3:
        array2D = array[0, :, :].squeeze()

    if type == 'opd':
        try:
            m = max(array2D.shape)
        except NameError:
            m = max(array.shape)
        k = 1
        imagedelta = fno * wlum
    else:
        m = max(pmask.shape)
        k = fno * wlum / imagedelta
        m = int(np.round(m*k))
        D = D*k

    mtfa = createMTFatm(D, m, k, wlum, zen, r0inmRef)

    if type == 'opd':
        try:
            iad = (array2D != 0)
        except NameError:
            iad = (array != 0)
    elif type == 'psf':
        mk = int(m + np.rint((m * (k - 1) + 1e-5) / 2) * 2)  # add even number
        iad = pmask  # padArray(pmask, m)
        
    # coordinates of the PSF in mas
    conv = 206265000. #=3600*180/pi*1000; const. for converting radian to mas
    da = conv*wl/D  #in arcsec; if type==psf, D includes the padding, so this is still valid
    ha = da*(m-1)/2
    ha1d = np.linspace(-ha, ha, m)
    xxr, yyr = np.meshgrid(ha1d, ha1d);

    # Perfect telescope
    opdt = np.zeros((m, m))
    psft = opd2psf(opdt, iad, wlum, imagedelta, 1, fno, debugLevel)
    otft = psf2otf(psft)  # OTF of perfect telescope
    otfa = otft * mtfa  # add atmosphere to perfect telescope
    psfa = otf2psf(otfa)

    dm=np.max(psfa)
    idxmax=(psfa==dm)
    idx = np.abs(psfa-0.5*dm) < fwhm_thresh*dm
    r = np.sqrt((xxr[idx]-xxr[idxmax])**2+(yyr[idx]-yyr[idxmax])**2)
    fwhmatm = 2*np.mean(r)

    # Error;
    if type == 'opd':
        if array.ndim == 2:
            ninst = 1
        else:
            ninst = array.shape[0]
        for i in range(ninst):
            if array.ndim == 2:
                array2D = array
            else:
                array2D = array[i, :, :].squeeze()
            psfei = opd2psf(array2D, iad, wlum, 0, 0, 0, debugLevel)
            if i == 0:
                psfe = psfei
            else:
                psfe += psfei
        psfe = psfe / ninst
    else:
        if array.shape[0] == mk:
            psfe = array
        elif array.shape[0] > mk:
            psfe = extractArray(array, mk)
        else:
            print('calc_pssn: image provided too small, %d < %d x %6.4f' % (
                array.shape[0], m, k))
            print('IQ is over-estimated !!!')
            psfe = padArray(array, mk)

        psfe = psfe / np.sum(psfe) * np.sum(psft)

    otfe = psf2otf(psfe)  # OTF of error
    otftot = otfe * mtfa  # add atmosphere to error
    psftot = otf2psf(otftot)

    dm=np.max(psftot)
    idxmax=(psftot==dm)
    idx = np.abs(psftot-0.5*dm) < fwhm_thresh*dm
    r = np.sqrt((xxr[idx]-xxr[idxmax])**2+(yyr[idx]-yyr[idxmax])**2)
    fwhmtot = 2*np.mean(r)
    
    fwhm_mas = np.max((0,(fwhmtot**power-fwhmatm**power)**(1/power))) #cannot be negative

    return fwhm_mas

