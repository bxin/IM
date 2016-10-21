#!/usr/bin/env python

# @author: Bo Xin
# @      Large Synoptic Survey Telescope

# main function

import argparse
# import numpy as np

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

    parser.add_argument('iSim', type=int, help='sim#')
    parser.add_argument('-icomp', type=int,
                        help='override icomp in the estimator parameter file, \
                        default=no override')
    parser.add_argument('-izn3', type=int,
                        help='override izn3 in the estimator parameter file, \
                        default=no override')
    parser.add_argument('-start', dest='startiter', type=int, default=0,
                        help='iteration No. to start with, default=0')
    parser.add_argument('-end', dest='enditer', type=int, default=5,
                        help='iteration No. to end with, default=5')
    parser.add_argument('-sensor', dest='sensor', choices = ('ideal','covM','phosim','cwfs','load'),
                        help='ideal: use true wavefront in estimator;\
                        covM: use covarance matrix to estimate wavefront;\
                        phosim: run Phosim to create WFS images;\
                        cwfs: start by running cwfs on existing images;\
                        load: load wavefront from txt files')
    parser.add_argument('-ctrloff', help='w/o applying ctrl rules or regenrating pert files',
                        action='store_true')
    parser.add_argument('-opdoff', help='w/o regenerating OPD maps',
                        action='store_true')
    parser.add_argument('-psfoff', help='w/o regenerating psf images',
                        action='store_true')
    parser.add_argument('-pssnoff', help='w/o calculating PSSN',
                        action='store_true')
    parser.add_argument('-ellioff', help='w/o calculating ellipticity',
                        action='store_true')
    parser.add_argument('-makesum', help='make summary plot, assuming all data available',
                        action='store_true')
    parser.add_argument('-p', dest='numproc', default=1, type=int,
                        help='Number of Processors Phosim uses')
    parser.add_argument('-g', dest='gain', default=0.7, type=float,
                        help='override gain in the controller parameter file, \
                        default=no override')
    parser.add_argument('-i', dest='inst',
                        default='lsst', #choices=('lsst','comcam'),
                        help='instrument name, \
                        default=lsst')
    parser.add_argument('-s', dest='simuParam',
                        default='single_dof',
                        help='simulation parameter file in data/, \
                        default=single_dof')
    parser.add_argument('-e', dest='estimatorParam',
                        default='pinv',
                        help='estimator parameter file in data/, default=pinv')
    parser.add_argument('-c', dest='controllerParam',
                        default='optiPSSN', choices=('optiPSSN', 'optiPSSN_0', 'null'),
                        help='controller parameter file in data/, \
                        default=optiPSSN')
    parser.add_argument('-w', dest='wavelength', type=float,
                        default=0.5, help='wavelength in micron, default=0.5')
    parser.add_argument('-d', dest='debugLevel', type=int,
                        default=0, choices=(-1, 0, 1, 2, 3),
                        help='debug level, -1=quiet, 0=Zernikes, \
                        1=operator, 2=expert, 3=everything, default=0')
    parser.add_argument('-baserun', dest='baserun', default=-1, type=int,
                        help='iter0 is same as this run, so skip iter0')
    args = parser.parse_args()
    if args.makesum:
        args.sensor = 'load'
        args.ctrloff = True
        args.opdoff = True
        args.psfoff = True
        args.pssnoff = True
        args.ellioff = True
        
    if args.debugLevel >= 1:
        print(args)

    # *****************************************
    # simulate the perturbations
    # *****************************************
    M1M3 = aosM1M3(args.debugLevel)
    M2 = aosM2(args.debugLevel)
    phosimDir = '../phosimSE/'
    #znPert = 28  # znmax used in pert file to define surfaces

    # *****************************************
    # run wavefront sensing algorithm
    # *****************************************
    cwfsDir = '../../wavefront/cwfs/'
    algoFile = 'exp'
    wfs = aosWFS(cwfsDir, args.inst, algoFile,
                 128, args.wavelength, args.debugLevel)

    cwfsModel = 'offAxis'

    # *****************************************
    # state estimator
    # *****************************************
    esti = aosEstimator(args.inst, args.estimatorParam, wfs, args.icomp, args.izn3,
                        args.debugLevel)
    # state is defined after esti, b/c, for example, ndof we use in state
    # depends on the estimator.
    pertDir = 'pert/sim%d' % args.iSim
    imageDir = 'image/sim%d' % args.iSim
    state = aosTeleState(args.inst, args.simuParam, args.iSim,
                         esti.ndofA, phosimDir,
                         pertDir, imageDir, args.debugLevel, M1M3=M1M3, M2=M2)
    # *****************************************
    # control algorithm
    # *****************************************
    metr = aosMetric(args.inst, state, wfs.znwcs3, args.debugLevel)
    ctrl = aosController(args.inst, args.controllerParam, esti, metr, wfs, M1M3, M2,
                         args.wavelength, args.gain, args.debugLevel)

    # *****************************************
    # start the Loop
    # *****************************************
    for iIter in range(args.startiter, args.enditer + 1):
        if args.debugLevel >= 3:
            print('iteration No. %d' % iIter)

        state.setIterNo(metr, iIter, wfs=wfs)

        if not args.ctrloff:
            if iIter > 0: #args.startiter:
                esti.estimate(state, wfs, ctrl, args.sensor)
                ctrl.getMotions(esti, metr, wfs, state, args.wavelength)
                ctrl.drawControlPanel(esti, state)

                # need to remake the pert file here.
                # It will be inserted into OPD.inst, PSF.inst later
                state.update(ctrl)

            state.writePertFile(esti.ndofA)

        if args.baserun>0 and iIter == 0:
            state.getOPDAllfromBase(args.baserun, metr)
            state.getPSFAllfromBase(args.baserun, metr)
            metr.getPSSNandMorefromBase(args.baserun, state)
            metr.getEllipticityfromBase(args.baserun, state)
            if (args.sensor == 'ideal' or args.sensor == 'covM' or args.sensor == 'load'):
                pass
            else:
                wfs.getZ4CfromBase(args.baserun, state)
        else:
            state.getOPDAll(args.opdoff, wfs, metr, args.numproc, args.wavelength,
                           args.debugLevel)

            state.getPSFAll(args.psfoff, metr, args.numproc, args.debugLevel)
    
            metr.getPSSNandMore(args.pssnoff, state, wfs, args.wavelength, args.numproc, args.debugLevel)
    
            metr.getEllipticity(args.ellioff, state, wfs, args.wavelength, args.numproc, args.debugLevel)
    
            if (args.sensor == 'ideal' or args.sensor == 'covM' or args.sensor == 'load'):
                pass
            else:
                if args.sensor == 'phosim':
                    # create donuts for last iter, so that picking up from there will be easy
                    state.getWFSAll(wfs, metr, args.numproc, args.debugLevel)
                    wfs.preprocess(state, metr, args.debugLevel)
                if args.sensor == 'phosim' or args.sensor == 'cwfs':                
                    wfs.parallelCwfs(cwfsModel, args.numproc, args.debugLevel)
                if args.sensor == 'phosim' or args.sensor == 'cwfs' or args.sensor == 'load':
                    wfs.checkZ4C(state, metr, args.debugLevel)

    ctrl.drawSummaryPlots(state, metr, esti, M1M3, M2,
                          args.startiter, args.enditer, args.debugLevel)

if __name__ == "__main__":
    main()
