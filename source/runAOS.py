#!/usr/bin/env python

# @author: Bo Xin
# @      Large Synoptic Survey Telescope

# main function
import matplotlib
matplotlib.use('Agg')
import argparse
# import numpy as np
import datetime
import os
import sys
import subprocess
import pytz

from aosWFS import aosWFS
from aosEstimator import aosEstimator
from aosController import aosController
from aosMetric import aosMetric
from aosM1M3 import aosM1M3
from aosM2 import aosM2
from aosTeleState import aosTeleState
from catalog import Catalog, GridCatalog


def main():
    date0 = datetime.datetime.now(pytz.timezone('America/Los_Angeles')).replace(microsecond=0)
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
    parser.add_argument('-sensor', dest='sensor',
                        choices=('ideal', 'covM', 'phosim', 'cwfs',
                                 'check', 'pass'),
                        help='ideal: use true wavefront in estimator;\
                        covM: use covarance matrix to estimate wavefront;\
                        phosim: run Phosim to create WFS images;\
                        cwfs: start by running cwfs on existing images;\
                        check: check wavefront against truth; \
                        pass: do nothing')
    parser.add_argument('-ctrloff', help='w/o applying ctrl rules or\
regenrating pert files',
                        action='store_true')
    parser.add_argument('-opdoff', help='w/o regenerating OPD maps',
                        action='store_true')
    parser.add_argument('-pssnoff', help='w/o calculating PSSN',
                        action='store_true')
    parser.add_argument('-ellioff', help='w/o calculating ellipticity',
                        action='store_true')
    parser.add_argument('-makesum', help='make summary plot,\
assuming all data available',
                        action='store_true')
    parser.add_argument('-p', dest='numproc', default=1, type=int,
                        help='Number of Processors Phosim uses')
    parser.add_argument('-g', dest='gain', default=0.7, type=float,
                        help='override gain in the controller parameter file, \
                        default=no override')
    parser.add_argument('-i', dest='inst',
                        default='lsst',  # choices=('lsst','comcam'),
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
                        default='optiPSSN_x0', choices=(
                            'optiPSSN_x0', 'optiPSSN_0', 'optiPSSN_x0xcor',
                            'optiPSSN_x00', 'null'),
                        help='controller parameter file in data/, \
                        default=optiPSSN')
    parser.add_argument('-w', dest='wavestr',
                        choices=('0.5', 'u', 'g', 'r', 'i', 'z', 'y'),
                        default="0.5", help='wavelength in micron, \
                        default=0.5')
    parser.add_argument('-d', dest='debugLevel', type=int,
                        default=0, choices=(-1, 0, 1, 2, 3),
                        help='debug level, -1=quiet, 0=Zernikes, \
                        1=operator, 2=expert, 3=everything, default=0')
    parser.add_argument('-o', dest='outputDir', default = '',
                        help='output directory,\
                        default=aosSrcDir/../')
    parser.add_argument('-baserun', dest='baserun', default=-1, type=int,
                        help='iter0 is same as this run, so skip iter0')
    args = parser.parse_args()
    if args.makesum:
        args.sensor = 'pass'
        args.ctrloff = True
        args.opdoff = True
        args.pssnoff = True
        args.ellioff = True

    if args.debugLevel >= 1:
        print(args)

    if args.wavestr == '0.5':
        band = 'g'
        wavelength = float(args.wavestr)
    else:
        band = args.wavestr
        wavelength = 0 #effwave[args.wavestr]

    aosSrcDir = os.path.split(os.path.abspath(__file__))[0]
    # *****************************************
    # simulate the perturbations
    # *****************************************
    M1M3 = aosM1M3(args.debugLevel)
    M2 = aosM2(args.debugLevel)
    phosimDir = '{}/../../phosim_syseng4'.format(aosSrcDir)
    pertDir = '%s/../pert/sim%d' %(aosSrcDir, args.iSim)
    if not args.outputDir:
        pertDir = '%s/../pert/sim%d' %(aosSrcDir, args.iSim)
        imageDir = '%s/../image/sim%d' %(aosSrcDir, args.iSim)
    else:
        pertDir = '%s/pert/sim%d' %(args.outputDir, args.iSim)
        imageDir = '%s/image/sim%d' %(args.outputDir, args.iSim)

    # *****************************************
    # run wavefront sensing algorithm
    # *****************************************
    cwfsDir = '{}/../../cwfs'.format(aosSrcDir)
    imDir = '{}/../../IM'.format(aosSrcDir)
    algoFile = 'exp'
    if wavelength == 0:
        effwave = aosTeleState.effwave[band]
    else:
        effwave = wavelength
    wfs = aosWFS(cwfsDir, imageDir, args.inst, algoFile, args.iSim, 192
                 , band, effwave, args.debugLevel)

    cwfsModel = 'offAxis'

    # *****************************************
    # state estimator
    # *****************************************
    esti = aosEstimator(args.inst, args.estimatorParam, wfs, args.icomp,
                        args.izn3, args.debugLevel)
    # state is defined after esti, b/c, for example, ndof we use in state
    # depends on the estimator.
    state = aosTeleState(args.inst, args.simuParam, args.iSim,
                         esti.ndofA, phosimDir,
                         pertDir, imageDir, band, wavelength,
                         args.enditer,
                         args.debugLevel, M1M3=M1M3, M2=M2)
    wfs.setIsr(state.eimage)
    # *****************************************
    # control algorithm
    # *****************************************
    metr = aosMetric(args.inst, state.opdSize, wfs.znwcs3, args.debugLevel)
    ctrl = aosController(args.inst, args.controllerParam, esti, metr, wfs,
                         M1M3, M2,
                         effwave, args.gain, args.debugLevel)

    catalog = GridCatalog()
    
    # catalog = Catalog()
    # d = 0.02
    # mag = state.cwfsMag
    # sed = state.sedfile
    # for i in range(metr.nField, metr.nFieldp4):
    #     x = metr.fieldXp[i]
    #     y = metr.fieldYp[i]
    #     if i % 2 == 1: # field 31, 33; R44 and R00
    #         catalog.addSource(x + d, y, mag, sed)
    #         catalog.addSource(x - d, y, mag, sed)
    #     else:
    #         catalog.addSource(x, y + d, mag, sed)
    #         catalog.addSource(x, y - d, mag, sed)

    # *****************************************
    # start the Loop
    # *****************************************
    for iIter in range(args.startiter, args.enditer + 1):
        if args.debugLevel >= 3:
            print('iteration No. %d' % iIter)

        state.setIterNo(metr, iIter, wfs=wfs)
        wfs.setIterNo(iIter)

        if not args.ctrloff:
            if iIter > 0:  # args.startiter:
                esti.estimate(state, wfs, ctrl, args.sensor)
                ctrl.getMotions(esti, metr, wfs, state)
                ctrl.drawControlPanel(esti, state)

                # need to remake the pert file here.
                # It will be inserted into OPD.inst, PSF.inst later
                state.update(esti, ctrl, M1M3, M2)
            if args.baserun > 0 and iIter == 0:
                state.getPertFilefromBase(args.baserun)
            else:
                state.writePertFile(esti.ndofA, M1M3=M1M3, M2=M2)

        if args.baserun > 0 and iIter == 0:
            state.getOPDAllfromBase(args.baserun, metr)
            metr.getPSSNandMorefromBase(args.baserun, state)
            metr.getEllipticityfromBase(args.baserun, state)
            if (args.sensor == 'ideal' or args.sensor == 'covM' or
                    args.sensor == 'pass' or args.sensor == 'check'):
                pass
            else:
                wfs.getZ4CfromBase(args.baserun, state)
        else:
            state.getOPDAll(args.opdoff, metr, args.numproc,
                            wfs.znwcs, wfs.inst.obscuration, args.debugLevel)

            metr.getPSSNandMore(args.pssnoff, state,
                                args.numproc, args.debugLevel)

            metr.getEllipticity(args.ellioff, state,
                                args.numproc, args.debugLevel)

            if (args.sensor == 'ideal' or args.sensor == 'covM' or
                    args.sensor == 'pass'):
                pass
            else:
                if args.sensor == 'phosim':
                    # create donuts for last iter,
                    # so that picking up from there will be easy
                    state.getWFSAll(wfs, catalog, args.numproc, args.debugLevel)
                    state.makeAtmosphereFile(metr, wfs, args.debugLevel)
                if args.sensor == 'phosim' or args.sensor == 'cwfs':
                    wfs.parallelCwfs(catalog, cwfsModel, args.numproc, args.debugLevel)
                if args.sensor == 'phosim' or args.sensor == 'cwfs' \
                        or args.sensor == 'check':
                    wfs.checkZ4C(state, metr, args.debugLevel)

    ctrl.drawSummaryPlots(state, metr, esti, M1M3, M2,
                          args.startiter, args.enditer, args.debugLevel)
    catalog.table.write('{}/catalog.csv'.format(pertDir), format='csv', overwrite=True)
    logRunInfo(os.path.join(pertDir, 'logRunInfo.txt'), cwfsDir, imDir, phosimDir, date0, args.startiter, args.enditer)

    print('Done runnng iterations: %d to %d' % (args.startiter, args.enditer))

def logRunInfo(path, cwfsDir, imDir, phosimDir, date0, startiter, enditer):
    date = datetime.datetime.now(pytz.timezone('America/Los_Angeles')).replace(microsecond=0)
    args = ' '.join(sys.argv)
    cmd = 'git -C {} log --pretty=format:"%h" -n 1'
    cwfsVersion = subprocess.check_output(cmd.format(cwfsDir), shell=True).decode('ascii')
    imVersion = subprocess.check_output(cmd.format(imDir), shell=True).decode('ascii')
    phosimVersion = subprocess.check_output(cmd.format(phosimDir), shell=True).decode(
        'ascii')
    log = """started: {}
finished: {}
iterations from {} to {}
average time per iteration: {}
args: {}
cwfs commit: {}
im commit: {}
phosim commit: {}
""".format(date0, date, startiter, enditer, (date-date0)/(enditer-startiter+1), args, cwfsVersion, imVersion, phosimVersion)
    with open(path, 'w') as fid:
        fid.write(log)

if __name__ == "__main__":
    main()
