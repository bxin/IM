#!/usr/bin/env python

# @author: Bo Xin
# @      Large Synoptic Survey Telescope

# main function

import argparse

from aosMetric import aosMetric
from aosTeleState import aosTeleState

def main():
    parser = argparse.ArgumentParser(
        description='-----Chromatic Validation------')
    parser.add_argument('iSim', type=int, help='sim#')
    parser.add_argument('-psfoff', help='w/o regenerating psf images',
                        action='store_true')
    parser.add_argument('-fftpsfoff', help='w/o calculating FFT psf images',
                        action='store_true')
    parser.add_argument('-pssnoff', help='w/o calculating PSSN',
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

    state = aosTeleState(inst, args.simuParam, args.iSim,
                         ndofA, phosimDir,
                         pertDir, imageDir, args.debugLevel)
    znwcs3 = 22 - 3
    metr = aosMetric(inst, state, znwcs3, args.debugLevel)

    nIter = 6
    wave = [0, 622, 550, 694, 586, 658]
    wlwt = [1, 1, 1, 1, ,1, 1]
    pixelum = 0.1 # 0.1um = 2mas
    
    for iIter in range(nIter):
        state.setIterNo(metr, iIter)
        state.writePertFile(ndofA)

        if iIter>0:
            state.getOPDAll(args.opdoff, wfs, metr, args.numproc,
                            args.wavelength, args.debugLevel)
            metr.getFFTPSF(args.fftpsfoff, state, wfs, args.wavelength,
                            args.numproc, args.debugLevel)
            
        state.getPSFAll(args.psfoff, metr, args.numproc, args.debugLevel,
                        pixelum=pixelum, wavelength = wave[iIter])

        if iIter>0:
            checkFFTPSF()
            metr.getPSSNandMore(args.pssnoff, state, wfs, args.wavelength,
                                    args.numproc, args.debugLevel, pixelum=pixelum)
            checkPSSN()
                 
        metr.getPSSNandMoreStamp(args.pssnoff, state, wfs, args.wavelength,
                                args.numproc, args.debugLevel, pixelum=pixelum)
        metr.getEllipticityStamp(args.ellioff, state, wfs, args.wavelength,
                                args.numproc, args.debugLevel, pixelum=pixelum)

    makeSumPlot()

def checkFFTPSF():
    """
    for a single wavelength, check Phosim fine-pixel PSF against Phosim OPD
    """
    pass

def checkPSSN():
    """
    for a single wavelength, check calc_PSSN(PSF) against calc_PSSN(OPD)
    """

    pass

def makeSumPlot():
    pass

if __name__ == "__main__":
    main()
