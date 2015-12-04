#!/usr/bin/env python

# @author: Bo Xin
# @      Large Synoptic Survey Telescope

# main function

import os
import argparse
import numpy as np

from aosWFS import aosWFS
from aosEstimator import aosEstimator
from aosController import aosController
from aosMetric import aosMetric
from aosM1M3 import aosM1M3
from aosM2 import aosM2
from aosTeleState import aosTeleState

def main():
    parser = argparse.ArgumentParser(
        description='-----LSST Integrated Model------')

    parser.add_argument('iSim',type=int,help='sim#')
    parser.add_argument('-icomp',type=int,
                        help='override icomp in the estimator parameter file, \
                        default=no override')
    parser.add_argument('-izn3',type=int,
                        help='override izn3 in the estimator parameter file, \
                        default=no override')
    parser.add_argument('-start',dest='startiter',type=int,default=0,
                        help='iteration No. to start with, default=0')
    parser.add_argument('-end',dest='enditer',type=int,default=5,
                        help='iteration No. to end with, default=5')
    parser.add_argument('-sensoroff',help='use true wavefront in estimator',
                        action='store_true')
    parser.add_argument('-opdoff',help='w/o regenerating OPD maps',
                        action='store_true')
    parser.add_argument('-psfoff',help='w/o regenerating psf images',
                        action='store_true')
    parser.add_argument('-wfsoff',help='w/o regenerating WFS images',
                        action='store_true')
    parser.add_argument('-pssnoff',help='w/o calculating PSSN',
                        action='store_true')
    parser.add_argument('-ellioff',help='w/o calculating ellipticity',
                        action='store_true')    
    parser.add_argument('-p',dest='numproc',default=1,type=int,
                        help='Number of Processors Phosim uses')    
    parser.add_argument('-g',dest='gain',default=0.7, type=float,
                        help='override gain in the controller parameter file, \
                        default=no override')
    parser.add_argument('-i', dest='instruParam',
                        default='single_dof',
                        help='instrument parameter file in data/, \
                        default=single_dof')
    parser.add_argument('-e', dest='estimatorParam',
                        default='pinv',
                        help='estimator parameter file in data/, default=pinv')
    parser.add_argument('-c', dest='controllerParam',
                        default='optiPSSN', choices=('optiPSSN','null'),
                        help='controller parameter file in data/, default=optiPSSN')
    parser.add_argument('-w',dest='wavelength',type=float,
                       default=0.5,help='wavelength in micron, default=0.5')
    parser.add_argument('-d', dest='debugLevel', type=int,
                        default=0, choices=(-1, 0, 1, 2, 3),
                        help='debug level, -1=quiet, 0=Zernikes, \
                        1=operator, 2=expert, 3=everything, default=0')
    args = parser.parse_args()
    if args.debugLevel >= 1:
        print(args)

    # *****************************************
    # simulate the perturbations
    # *****************************************
    M1M3=aosM1M3(args.debugLevel)
    M2 =aosM2(args.debugLevel)
    phosimDir='../phosimSE/'
    znPert=28 #znmax used in pert file to define surfaces
        
    # *****************************************
    # run wavefront sensing algorithm
    # *****************************************
    cwfsDir='../../wavefront/cwfs/'
    instruFile='lsst'
    algoFile='exp'
    imgSizeinPix=128
    wfs=aosWFS(cwfsDir,instruFile,algoFile,
               128,args.wavelength,args.debugLevel)

    cwfsInstru = 'lsst'
    cwfsAlgo='exp'
    cwfsModel = 'offAxis'
    
    # *****************************************
    # state estimator
    # *****************************************
    esti=aosEstimator(args.estimatorParam, args.icomp,args.izn3,
                       args.debugLevel)
    #state is defined after esti, b/c, for example, ndof we use in state
    # depends on the estimator.
    pertDir='pert/sim%d'%args.iSim
    if not os.path.isdir(pertDir):
        os.mkdir(pertDir)
    imageDir='image/sim%d'%args.iSim
    if not os.path.isdir(imageDir):
        os.mkdir(imageDir)
    state=aosTeleState(esti, args.instruParam, args.iSim, phosimDir,
                       pertDir, imageDir, args.debugLevel)
    # *****************************************
    # control algorithm
    # *****************************************
    metr=aosMetric(wfs, args.debugLevel)
    ctrl=aosController(args.controllerParam, esti, metr, M1M3, M2,
                       args.wavelength, args.gain, args.debugLevel)
        
    # *****************************************    
    # start the Loop
    # *****************************************
    for iIter in range(args.startiter, args.enditer+1):
        if args.debugLevel>=3:
            print('iteration No. %d'%iIter)

        state.setIterNo(iIter)
    
        if iIter>args.startiter:
            esti.estimate(state, wfs, args.sensoroff)
            ctrl.getMotions(esti, state)

            #need to remake the pert file here.
            #It will be inserted into OPD.inst, PSF.inst later
            state.update(ctrl)
                        
            exit()
    
        state.writePertFile(esti)
        if not args.opdoff:
            state.getOPD35(wfs, metr, args.numproc, args.wavelength,
                           args.debugLevel)
        if not args.psfoff:
            state.getPSF31(metr, args.numproc, args.debugLevel)

        if not args.pssnoff:
            metr.getPSSNandMore(state, args.wavelength, args.debugLevel)
            
        if not args.ellioff:
            metr.getEllipticity(state, args.wavelength, args.debugLevel)

        if not args.sensoroff:
            if not args.wfsoff: # and not iIter == args.enditer:
                state.getWFS4(metr, args.numproc, args.debugLevel)
            #aosWFS

                        
if __name__ == "__main__":
    main()
