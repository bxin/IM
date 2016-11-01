#!/usr/bin/env python

# @author: Bo Xin
# @      Large Synoptic Survey Telescope

# main function

import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from aosMetric import aosMetric
from aosTeleState import aosTeleState

from lsst.cwfs.tools import extractArray

def main():
    parser = argparse.ArgumentParser(
        description='-----Chromatic Validation------')
    parser.add_argument('iSim', type=int, help='sim#')
    parser.add_argument('-opdoff', help='w/o regenerating OPD maps',
                        action='store_true')
    parser.add_argument('-psfoff', help='w/o regenerating psf images',
                        action='store_true')
    parser.add_argument('-fftpsfoff', help='w/o calculating FFT psf images',
                        action='store_true')
    parser.add_argument('-pssnoff', help='w/o calculating PSSN',
                        action='store_true')
    parser.add_argument('-ellioff', help='w/o calculating ellipticity',
                        action='store_true')
    parser.add_argument('-p', dest='numproc', default=1, type=int,
                        help='Number of Processors Phosim uses')
    parser.add_argument('-s', dest='simuParam',
                        default='single_dof',
                        help='simulation parameter file in data/, \
                        default=single_dof')
    parser.add_argument('-b', dest='band',default = 'r',
                        choices=('u','g','r','i','z','y'),
                        help='optical band, default = r')
    parser.add_argument('-d', dest='debugLevel', type=int,
                        default=0, choices=(-1, 0, 1, 2, 3),
                        help='debug level, -1=quiet, 0=Zernikes, \
                        1=operator, 2=expert, 3=everything, default=0')
    args = parser.parse_args()

    inst = 'lsst'
    ndofA =50
    phosimDir = '../phosimSE/'
    pertDir = 'pert/sim%d' % args.iSim
    imageDir = 'image/sim%d' % args.iSim

    znwcs = 22
    znwcs3 = znwcs - 3
    obscuration = 0.61

    nIter = 6
    band = 'r'
    wave = [0.622, 0.550, 0.694, 0.586, 0.658, 0]
    wlwt = [1, 1, 1, 1, 1, 1]
    pixelum = 0.2 # 0.1um = 2mas
    
    # for iIter in range(nIter):
    for iIter in range(1):
        wavelength = wave[iIter]
            
        state = aosTeleState(inst, args.simuParam, args.iSim,
                            ndofA, phosimDir,
                            pertDir, imageDir, band, wavelength,
                                 args.debugLevel)
        metr = aosMetric(inst, state.opdSize, znwcs3, args.debugLevel,
                             pixelum=pixelum)
        
        state.setIterNo(metr, iIter)
        state.writePertFile(ndofA)

        #a non-zero pixelum indicates we want to use fine-pixel psf images
        state.getPSFAll(args.psfoff, metr, args.numproc, args.debugLevel,
                        pixelum=pixelum)

        metr.getPSSNandMore(args.pssnoff, state, wavelength,
                                args.numproc, znwcs, obscuration,
                                     args.debugLevel, pixelum=pixelum)
        metr.getEllipticity(args.ellioff, state, wavelength,
                                args.numproc, znwcs, obscuration, 
                                     args.debugLevel, pixelum=pixelum)
        
        if iIter< nIter-1:
            state.getOPDAll(args.opdoff, metr, args.numproc, wavelength,
                                znwcs, obscuration, args.debugLevel)
            metr.getFFTPSF(args.fftpsfoff, state, wavelength, pixelum,
                            args.numproc, args.debugLevel)
            
            checkPSF(metr, state, 2)
            checkPSF(metr, state, 1)
            # below, pixelum uses default value 0, opd maps will be used
            metr.getPSSNandMore(args.pssnoff, state, wavelength,
                                    args.numproc, znwcs, obscuration,
                                    args.debugLevel, outFile =
                                    metr.PSSNFile.replace(
                                        'PSSN.txt','opdPSSN.txt'))
            # below, pixelum < 0, fftpsf will be used
            metr.getPSSNandMore(args.pssnoff, state, wavelength,
                                    args.numproc, znwcs, obscuration,
                                    args.debugLevel, outFile =
                                    metr.PSSNFile.replace(
                                        'PSSN.txt','fftpsfPSSN.txt'),
                                    pixelum = -pixelum) #use fftpsf
            # below, pixelum uses default value 0, opd maps will be used
            metr.getEllipticity(args.ellioff, state, wavelength,
                                    args.numproc, znwcs, obscuration, 
                                    args.debugLevel, outFile =
                                    metr.elliFile.replace(
                                        'elli.txt','opdElli.txt'))
            # below, pixelum < 0, fftpsf will be used
            metr.getEllipticity(args.ellioff, state, wavelength,
                                    args.numproc, znwcs, obscuration, 
                                    args.debugLevel, outFile =
                                    metr.elliFile.replace(
                                        'elli.txt','fftpsfElli.txt'),
                                        pixelum = -pixelum) #use fftpsf 
            checkPSSN(metr, state)
            checkEllipticity(metr, state)
                 

    makeSumPlot()

def checkPSF(metr, state, dim):
    """
    for a single wavelength, check Phosim fine-pixel PSF against Phosim OPD
    """
    plt.figure(figsize=(10, 10))
    for i in range(metr.nField):    
        psfFile = '%s/iter%d/sim%d_iter%d_fftpsf%d.fits' % (
            state.imageDir, state.iIter, state.iSim, state.iIter, i)
        IHDU = fits.open(psfFile)
        fftpsf = IHDU[0].data
        IHDU.close()
        fftpsf = fftpsf/np.max(fftpsf)
        offsety = int(np.argwhere(fftpsf == fftpsf.max())[0][0] - \
          state.psfStampSize/2  + 1)
        offsetx = int(np.argwhere(fftpsf == fftpsf.max())[0][1] - \
          state.psfStampSize/2  + 1)
        fftpsf = np.roll(fftpsf, -offsety, axis = 0)
        fftpsf = np.roll(fftpsf, -offsetx, axis = 1)
        
        psfFile = '%s/iter%d/sim%d_iter%d_psf%d.fits' % (
            state.imageDir, state.iIter, state.iSim, state.iIter, i)
        IHDU = fits.open(psfFile)
        psf = IHDU[0].data
        IHDU.close()
        psf = psf/np.max(psf)
        
        if state.inst[:4] == 'lsst':
            if i == 0:
                pIdx = 1
            else:
                pIdx = i + metr.nArm
            nRow = metr.nRing + 1
            nCol = metr.nArm
        elif state.inst[:6] == 'comcam':
            aa = [7, 4, 1, 8, 5, 2, 9, 6, 3]
            pIdx = aa[i]
            nRow = 3
            nCol = 3
        displaySize = 100
        ax = plt.subplot(nRow, nCol, pIdx)
        if dim == 2:
            plt.imshow(extractArray(psf-fftpsf, displaySize),
                    origin='lower', interpolation='none')
            plt.clim(-1, 1)
            # plt.colorbar()
            plt.axis('off')
            
        elif dim == 1:
            x = range(state.psfStampSize)
            z1 = psf[state.psfStampSize/2-1,:]
            z2 = fftpsf[state.psfStampSize/2-1,:]                
            plt.plot(x, z1, label = 'psf', color='r', linewidth=0.5)
            plt.plot(x, z2, label = 'fftpsf', color='b', linewidth=0.5)
            ax.set_xticklabels([])
            ax.set_yticklabels([0, '', '', '', '', 1])
            leg = ax.legend(loc="upper right", fontsize=6)
            leg.get_frame().set_alpha(0.5)
            
        plt.title('Field %d' % i, fontsize=8)

    # plt.show()
    pngFile = '%s/iter%d/sim%d_iter%d_checkpsf_%dD.png' % (
        state.imageDir, state.iIter, state.iSim, state.iIter, dim)
    plt.savefig(pngFile, bbox_inches='tight', dpi=500)
    plt.close()
            
def checkPSSN(metr, state):
    """
    for a single wavelength, check calc_PSSN(PSF) against calc_PSSN(OPD)
    """
    plt.figure(figsize=(10, 6))
    x= range(metr.nField)
    z1 = np.loadtxt(metr.PSSNFile)
    z2 = np.loadtxt(metr.PSSNFile.replace('PSSN.txt','opdPSSN.txt'))
    z3 = np.loadtxt(metr.PSSNFile.replace('PSSN.txt','fftpsfPSSN.txt'))
    plt.subplot(1,2,1)
    plt.plot(x, z1[0, :metr.nField],  label = 'psf', marker = 'o', color='r')
    plt.plot(x, z2[0, :metr.nField],  label = 'opd', marker = 'x', color='b')
    plt.plot(x, z3[0, :metr.nField],  label = 'fftpsf', marker = '*', color='k')    
    leg = plt.legend(loc="upper right", fontsize=10)
    leg.get_frame().set_alpha(0.5)    
    plt.grid()
    plt.xlabel('Field Index')
    plt.ylabel('PSSN')
    
    plt.subplot(1,2,2)
    plt.plot(x, z1[1, :metr.nField],  label = 'psf', marker = 'o', color='r')
    plt.plot(x, z2[1, :metr.nField],  label = 'opd', marker = 'x', color='b')
    plt.plot(x, z3[1, :metr.nField],  label = 'fftpsf', marker = '*', color='k')    
    leg = plt.legend(loc="upper right", fontsize=10)
    leg.get_frame().set_alpha(0.5)
    plt.grid()
    plt.xlabel('Field Index')
    plt.ylabel('FWHMeff (arcsec)')
    
    # plt.show()
    pngFile = '%s/iter%d/sim%d_iter%d_checkPSSN.png' % (
        state.imageDir, state.iIter, state.iSim, state.iIter)
    plt.savefig(pngFile, bbox_inches='tight')
    plt.close()

def checkEllipticity(metr, state):
    """
    for a single wavelength, check elli(PSF) against elli(OPD)
    """

    plt.figure(figsize=(6, 6))
    x= range(metr.nField)
    z1 = np.loadtxt(metr.elliFile)
    z2 = np.loadtxt(metr.elliFile.replace('elli.txt','opdElli.txt'))
    z3 = np.loadtxt(metr.elliFile.replace('elli.txt','fftpsfElli.txt'))    
    plt.plot(x, z1[:metr.nField],  label = 'psf', marker = 'o', color='r')
    plt.plot(x, z2[:metr.nField],  label = 'opd', marker = 'x', color='b')
    plt.plot(x, z3[:metr.nField],  label = 'fftpsf', marker = '*', color='k')
    plt.legend(loc="upper right", fontsize=8)
    plt.grid()
    plt.xlabel('Field Index')
    plt.ylabel('elli')
    
    # plt.show()
    pngFile = '%s/iter%d/sim%d_iter%d_checkElli.png' % (
        state.imageDir, state.iIter, state.iSim, state.iIter)
    plt.savefig(pngFile, bbox_inches='tight')
    plt.close()

def makeSumPlot():
    """
    check the GQ sum of pssn/fwhm/fwhmeff/elli vs band measurements
    """
    pass

if __name__ == "__main__":
    main()
