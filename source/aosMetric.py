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

from cwfsTools import padArray
from cwfsTools import ZernikeAnnularFit
from cwfsTools import ZernikeAnnularEval

from cwfsErrors import nonSquareImageError
from aosErrors import psfSamplingTooLowError

# import cwfsPlots as plot


class aosMetric(object):

    def __init__(self, state, wfs, debugLevel):
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
        self.w = self.w / np.sum(self.w)
        # self.fieldX[self.nField:]=[1.185, -1.185, -1.185, 1.185]
        # self.fieldY[self.nField:]=[1.185, 1.185, -1.185, -1.185]
        # counter-clock wise
        self.fieldX[self.nField:] = [1.176, -1.176, -1.176, 1.176]
        self.fieldY[self.nField:] = [1.176, 1.176, -1.176, -1.176]

        # below, p is for PSF
        self.fieldXp = self.fieldX.copy()
        self.fieldYp = self.fieldY.copy()

        self.fieldXp[19] += 0.004
        self.fieldXp[22] -= 0.004

        self.fwhmModelFileBase = 'data/fwhmModel/fwhm_vs_z_500nm'

        if debugLevel >= 3:
            print(self.w.shape)
            print(self.w)

        aa = np.loadtxt('data/pssn_alpha.txt')
        self.pssnAlpha = aa[:, 0]
        # self.pssnRange = aa[: 1]

        self.znx2 = np.zeros((self.nFieldp4, wfs.znwcs3))
        self.stampD = 2**np.ceil(np.log2(state.opdSize))

    def getFWHMfromZ(self):
        self.fwhm = np.zeros(self.nField)

    def getPSSNfromZ(self):
        pass

    def getPSSNandMore(self, pssnoff, state, wfs, wavelength, numproc, debugLevel):

        if not pssnoff:
            # multithreading on MacOX doesn't work with pinv
            if sys.platform == 'darwin':
                self.PSSN = np.zeros(self.nField)
            argList = []
            for i in range(self.nField):
                opdFile = '%s/iter%d/sim%d_iter%d_opd%d.fits' % (
                    state.imageDir, state.iIter, state.iSim, state.iIter, i)
    
                argList.append((opdFile, state, wfs.znwcs,
                                wfs.inst.obscuration, wavelength, self.stampD,
                                debugLevel))
    
                # test, pdb cannot go into the subprocess
                # aa = runEllipticity(argList[0])
                if sys.platform == 'darwin':
                    self.PSSN[i] = runPSSNandMore(argList[i])
    
            # tested, but couldn't figure out why the below didn't work
            if sys.platform != 'darwin':
                pool = multiprocessing.Pool(numproc)
                self.PSSN = pool.map(runPSSNandMore, argList)
                pool.close()
                pool.join()
                self.PSSN = np.array(self.PSSN)
                
            self.FWHMeff = 1.086*0.6*np.sqrt(1/self.PSSN-1)
            self.dm5 = -1.25 * np.log10(self.PSSN)
    
            if debugLevel >= 2:
                for i in range(self.nField):
                    print('---field#%d, PSSN=%7.4f, FWHMeff = %5.0f mas' % (
                        i, self.PSSN[i], self.FWHMeff[i]*1e3))
    
            self.GQPSSN = np.sum(self.w * self.PSSN)
            self.GQFWHMeff = np.sum(self.w * self.FWHMeff)
            self.GQdm5 = np.sum(self.w * self.dm5)
            a1=np.concatenate((self.PSSN, self.GQPSSN*np.ones(1)))
            a2=np.concatenate((self.FWHMeff, self.GQFWHMeff*np.ones(1)))
            a3=np.concatenate((self.dm5, self.GQdm5*np.ones(1)))
            np.savetxt(self.PSSNFile, np.vstack((a1,a2,a3)))
            
            if debugLevel >= 2:
                print(self.GQPSSN)
        else:
            aa = np.loadtxt(self.PSSNFile)
            self.GQFWHMeff = aa[1, -1] #needed for shiftGear
            
    def getPSSNandMorefromBase(self, baserun, state):
        if not os.path.isfile(self.PSSNFile):        
            baseFile = self.PSSNFile.replace('sim%d'%state.iSim, 'sim%d'%baserun)
            os.link(baseFile, self.PSSNFile)
        aa = np.loadtxt(self.PSSNFile)
        self.GQFWHMeff = aa[1, -1] #needed for shiftGear
                    
    def getPSSNandMore10um(self, state, wavelength, debugLevel):
        """
use the Phosim PSFs with 10um pixel size to determine PSSN and more
to be implemented
        """
        self.PSSN = np.zeros(self.nField)
        self.FWHMeff = np.zeros(self.nField)
        self.dm5 = np.zeros(self.nField)

        for i in range(self.nField):
            opdFile = '%s/iter%d/sim%d_iter%d_opd%d.fits' % (
                state.imageDir, state.iIter, state.iSim, state.iIter, i)
            IHDU = fits.open(opdFile)
            opd = IHDU[0].data  #  um
            IHDU.close()

            if self.stampD > opd.shape[0]:
                a = opd
                opd = np.zeros((self.stampD, self.stampD))
                opd[:a.shape[0], :a.shape[1]] = a

            self.PSSN[i] = calc_pssn(opd, wavelength, debugLevel=debugLevel)
            self.FWHMeff[i] = 1.086*0.6*np.sqrt(1/self.PSSN[i]-1)
            self.dm5[i] = -1.25 * np.log10(self.PSSN[i])

            if debugLevel >= 2:
                print('---field#%d, PSSN=%7.4f, FWHMeff = %5.0f mas' % (
                    i, self.PSSN[i], self.FWHMeff[i]*1e3))

        self.GQPSSN = np.sum(self.w * self.PSSN)
        if debugLevel >= 2:
            print(self.GQPSSN)

    def getEllipticity(self, ellioff, state, wfs, wavelength, numproc, debugLevel):

        if not ellioff:
            # multithreading on MacOX doesn't work with pinv
            if sys.platform == 'darwin':
                self.elli = np.zeros(self.nField)
            argList = []
            for i in range(self.nField):
                opdFile = '%s/iter%d/sim%d_iter%d_opd%d.fits' % (
                    state.imageDir, state.iIter, state.iSim, state.iIter, i)
    
                argList.append((opdFile, state, wfs.znwcs,
                                wfs.inst.obscuration, wavelength, self.stampD,
                                debugLevel))
    
                # test, pdb cannot go into the subprocess
                # aa = runEllipticity(argList[0])
                if sys.platform == 'darwin':
                    self.elli[i] = runEllipticity(argList[i])
    
            # tested, but couldn't figure out why the below didn't work
            if sys.platform != 'darwin':
                pool = multiprocessing.Pool(numproc)
                self.elli = pool.map(runEllipticity, argList)
                pool.close()
                pool.join()
                
            for i in range(self.nField):
                if debugLevel >= 2:
                    print('---field#%d, elli=%7.4f' % (i, self.elli[i]))
    
            self.GQelli = np.sum(self.w * self.elli)
            a1=np.concatenate((self.elli, self.GQelli*np.ones(1)))
            np.savetxt(self.elliFile, a1)
            if debugLevel >= 2:
                print(self.GQelli)

    def getEllipticityfromBase(self, baserun, state):
        if not os.path.isfile(self.elliFile):        
            baseFile = self.elliFile.replace('sim%d'%state.iSim, 'sim%d'%baserun)
            os.link(baseFile, self.elliFile)

    def getEllipticity10um(self, state, wavelength, debugLevel):
        self.elli = np.zeros(self.nField)
        for i in range(self.nField):
            opdFile = '%s/iter%d/sim%d_iter%d_opd%d.fits' % (
                state.imageDir, state.iIter, state.iSim, state.iIter, i)
            IHDU = fits.open(opdFile)
            opd = IHDU[0].data  # um
            IHDU.close()

            if self.stampD > opd.shape[0]:
                a = opd
                opd = np.zeros((self.stampD, self.stampD))
                opd[:a.shape[0], :a.shape[1]] = a

            self.elli[i], _, _, _ = psf2eAtmW(
                opd, wavelength, debugLevel=debugLevel)

            if debugLevel >= 2:
                print('---field#%d, elli=%7.4f' % (i, self.elli[i]))
            # exit()

        self.GQelli = np.sum(self.w * self.elli)
        if debugLevel >= 2:
            print(self.GQelli)


def calc_pssn(array, wlum, type='opd', D=8.36, r0inmRef=0.1382, zen=0,
              pmask=0, imagedelta=0.2, fno=1.2335, debugLevel=0):
    """
    array: the array that contains eitehr opd or pdf
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
    size as array
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
        iad = padArray(pmask, m)

    # number of non-zero elements, used for normalization later
    # miad2 = np.count_nonzero(iad)

    # Perfect telescope
    opdt = np.zeros((m, m))
    psft = opd2psf(opdt, iad, wlum, 0, 0, 0, debugLevel)
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
        psfe = padArray(array, m)
        psfe = psfe / np.sum(psfe) * np.sum(psft)

    otfe = psf2otf(psfe)  # OTF of error
    otftot = otfe * mtfa  # add atmosphere to error
    psftot = otf2psf(otftot)
    pss = np.sum(psftot**2)  # atmospheric + error PSS

    pssn = pss / pssa  # normalized PSS

    return pssn


def createMTFatm(D, m, k, wlum, zen, r0inmRef):
    """
    m is the number of pixel we want to have to cover the length of D/wl.
    If we want a k-times bigger array, we pad the mtf generated using k=1.
    """

    sfa = atmSF('vonK', D, m, wlum, zen, r0inmRef)
    mtfa = np.exp(-0.5 * sfa)

    N = np.rint(m * k + 1e-5)
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
        midp = np.rint(0.5 * (r.shape[0] - 1) + 1e-2)
        sfa[midp, midp] = 0  # at this single point, sfa_k=Inf, 0*Inf=Nan;

    return sfa


def r0Wz(r0inmRef, zen, wlum):
    zen = zen * np.pi / 180.  # telescope zenith angle, change here
    r0aref = r0inmRef * np.cos(zen)**0.6  # atmosphere reference r0
    r0a = r0aref * (wlum / 0.5)**1.2  # atmosphere r0, a function of wavelength
    return r0a


def psf2eAtmW(wfm, wlum, D=8.36, pmask=0, r0inmRef=0.1382,
              sensorFactor=1,
              zen=0, imagedelta=0.2, fno=1.2335, debugLevel=0):
    """
    wfm: wavefront OPD in micron
    """
    psfe = opd2psf(wfm, 0, wlum, imagedelta, sensorFactor, fno, debugLevel)
    otfe = psf2otf(psfe)  # OTF of error

    m = wfm.shape[0] / sensorFactor
    k = fno * wlum / imagedelta
    # since padding=k/sensorFactor<=k, any psfSamplingTooLowError
    # would have been raised in opd2psf()
    mtfa = createMTFatm(D, m, k, wlum, zen, r0inmRef)

    otf = otfe * mtfa
    psf = otf2psf(otf)

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
        N = np.rint(padding * sensorSamples)
        pupil = padArray(pupil, N)
        opd = padArray(opd, N)
        if debugLevel >= 3:
            print('padding=%8.6f' % padding)

    z = pupil * np.exp(-2j * np.pi * opd / wavelength)
    z = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(z),
                                    s=z.shape))  # /sqrt(miad2/m^2)
    z = np.absolute(z**2)
    z = z / np.sum(z)

    if debugLevel >= 3:
        print('opd2psf(): imagedelta=%8.6f' % imagedelta)
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
    opdFile = argList[0]
    opdx = argList[1].opdx
    opdy = argList[1].opdy
    znwcs = argList[2]
    obsR = argList[3]
    wavelength = argList[4]
    stampD = argList[5]
    debugLevel = argList[6]
    print('runEllipticity: %s '% opdFile)
    
    IHDU = fits.open(opdFile)
    opd = IHDU[0].data # um
    IHDU.close()
    
    # before psf2eAtmW()
    # (1) remove PTT,
    # (2) make sure outside of pupil are all zeros
    idx = (opd != 0)
    
    Z = ZernikeAnnularFit(opd[idx], opdx[idx], opdy[idx], znwcs, obsR)    
    Z[3:] = 0
    opd[idx] -= ZernikeAnnularEval(Z, opdx[idx], opdy[idx], obsR)

    if stampD > opd.shape[0]:
        a = opd
        opd = np.zeros((stampD, stampD))
        opd[:a.shape[0], :a.shape[1]] = a

    elli, _, _, _ = psf2eAtmW(
        opd, wavelength, debugLevel=debugLevel)
    return elli

def runPSSNandMore(argList):
    opdFile = argList[0]
    opdx = argList[1].opdx
    opdy = argList[1].opdy
    znwcs = argList[2]
    obsR = argList[3]
    wavelength = argList[4]
    stampD = argList[5]
    debugLevel = argList[6]
    print('runPSSNandMore: %s '% opdFile)
    
    IHDU = fits.open(opdFile)
    opd = IHDU[0].data # unit: um
    IHDU.close()

    # before calc_pssn,
    # (1) remove PTT,
    # (2) make sure outside of pupil are all zeros
    idx = (opd != 0)
    Z = ZernikeAnnularFit(opd[idx], opdx[idx], opdy[idx], znwcs, obsR)
    Z[3:] = 0
    opd[idx] -= ZernikeAnnularEval(Z, opdx[idx], opdy[idx], obsR)

    if stampD > opd.shape[0]:
        a = opd
        opd = np.zeros((stampD, stampD))
        opd[:a.shape[0], :a.shape[1]] = a

    pssn = calc_pssn(opd, wavelength, debugLevel=debugLevel)
    return pssn

