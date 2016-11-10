#!/usr/bin/env python

# @author: Bo Xin
# @      Large Synoptic Survey Telescope

import os
import sys
import shutil
import glob
import subprocess
import multiprocessing

import numpy as np
from astropy.io import fits
import aosCoTransform as ct

from lsst.cwfs.tools import ZernikeAnnularFit
from lsst.cwfs.tools import ZernikeAnnularEval
from lsst.cwfs.tools import extractArray

import matplotlib.pyplot as plt

phosimFilterID = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'y': 5}


class aosTeleState(object):

    effwave = {'u': 0.365, 'g': 0.480, 'r': 0.622,
               'i': 0.754, 'z': 0.868, 'y': 0.973}
    GQwave = {'u': [0.365], # place holder
              'g': [0.41826, 0.44901, 0.49371, 0.53752],
              'r': [0.55894, 0.59157, 0.63726, 0.68020],
              'i': [0.70381, 0.73926, 0.78798, 0.83279],
              'z': [0.868], # place holder
              'y': [0.973]} # place holder
    GQwt = {'u': [1],
            'g': [0.14436, 0.27745, 0.33694, 0.24125],
            'r': [0.15337, 0.28587, 0.33241, 0.22835],
            'i': [0.15810, 0.29002, 0.32987, 0.22201],
            'z': [1],
            'y': [1]}
        
    def __init__(self, inst, instruFile, iSim, ndofA, phosimDir,
                 pertDir, imageDir, band, wavelength, debugLevel,
                 M1M3=None, M2=None):

        self.band = band
        self.wavelength = wavelength
        if wavelength == 0:
            self.effwave = aosTeleState.effwave[band]
            self.nOPDrun = len(aosTeleState.GQwave[band])
        else:
            self.effwave = wavelength
            self.nOPDrun = 1
        
        assert sum(aosTeleState.GQwt[self.band])-1 < 1e-3
        
        # plan to write these to txt files. no columns for iter
        self.stateV = np.zeros(ndofA)  # *np.nan # telescope state(?)

        aa = inst
        if aa[-2:].isdigit():
            aa = aa[:-2]
        self.inst = aa
        self.instruFile = os.path.join('data/', (instruFile + '.inst'))
        fid = open(self.instruFile)
        iscomment = False
        for line in fid:
            line = line.strip()
            if (line.startswith('###')):
                iscomment = ~iscomment
            if (not(line.startswith('#')) and
                    (not iscomment) and len(line) > 0):
                if (line.startswith('dof')):
                    self.stateV[int(line.split()[1]) -
                                1] = float(line.split()[2])
                    # by default we want micron and arcsec for everything
                    if line.split()[3] == 'mm':
                        self.stateV[int(line.split()[1]) - 1] *= 1e3
                    elif line.split()[3] == 'deg':
                        self.stateV[int(line.split()[1]) - 1] *= 3600
                elif (line.startswith('budget')):
                    # read in in mas, convert to arcsec
                    self.budget = np.sqrt(
                        np.sum([float(x)**2 for x in line.split()[1:]])) * 1e-3
                elif (line.startswith('zenithAngle')):
                    self.zAngle = float(line.split()[1]) / 180 * np.pi
                elif (line.startswith('camTB')):
                    self.camTB = float(line.split()[1])
                elif (line.startswith('camRotation')):
                    self.camRot = float(line.split()[1])
                elif (line.startswith('M1M3ForceError')):
                    self.M1M3ForceError = float(line.split()[1])
                elif (line.startswith('M1M3TxGrad')):
                    self.M1M3TxGrad = float(line.split()[1])
                elif (line.startswith('M1M3TyGrad')):
                    self.M1M3TyGrad = float(line.split()[1])
                elif (line.startswith('M1M3TzGrad')):
                    self.M1M3TzGrad = float(line.split()[1])
                elif (line.startswith('M1M3TrGrad')):
                    self.M1M3TrGrad = float(line.split()[1])
                elif (line.startswith('M1M3TBulk')):
                    self.M1M3TBulk = float(line.split()[1])
                elif (line.startswith('M2TzGrad')):
                    self.M2TzGrad = float(line.split()[1])
                elif (line.startswith('M2TrGrad')):
                    self.M2TrGrad = float(line.split()[1])
                elif (line.startswith('opd_size')):
                    self.opdSize = int(line.split()[1])
                    if self.opdSize % 2 == 0:
                        self.opdSize -= 1
                elif (line.startswith('eimage')):
                    self.eimage = bool(int(line.split()[1]))
                elif (line.startswith('psf_mag')):
                    self.psfMag = int(line.split()[1])
                elif (line.startswith('cwfs_mag')):
                    self.cwfsMag = int(line.split()[1])
                elif (line.startswith('cwfs_stamp_size')):
                    self.cwfsStampSize = int(line.split()[1])

        fid.close()

        self.fno = 1.2335
        k = self.fno * self.effwave / 0.2
        self.psfStampSize = int(self.opdSize +
                                np.rint((self.opdSize * (k - 1) + 1e-5) / 2) *
                                2)
        self.iSim = iSim
        self.phosimDir = phosimDir
        self.pertDir = pertDir
        # if not os.path.isdir(pertDir):
        #     os.makedirs(pertDir)
        self.imageDir = imageDir
        # if not os.path.isdir(imageDir):
        #     os.makedirs(imageDir)

        # self.setIterNo(0)
        self.phosimActuatorID = [
            # M2 z, x, y, rx, ry
            5, 6, 7, 8, 9,
            # Cam z, x, y, rx, ry
            10, 11, 12, 13, 14] + [
            # M13 and M2 bending
            i for i in range(15, 15 + ndofA - 10)]

        self.opdGrid1d = np.linspace(-1, 1, self.opdSize)
        self.opdx, self.opdy = np.meshgrid(self.opdGrid1d, self.opdGrid1d)
        # runProgram('rm -rf %s/output/*'%self.phosimDir)

        if self.wavelength == 0:
            self.sedfile = 'sed_flat.txt'
        else:
            self.sedfile = 'sed_%d.txt' % (self.wavelength * 1e3)
            sedfileFull = '%s/data/sky/%s' % (self.phosimDir, self.sedfile)
            if not os.path.isfile(sedfileFull):
                fsed = open(sedfileFull, 'w')
                fsed.write('%d   1.0\n' % (self.wavelength * 1e3))
                fsed.close()
        
        if debugLevel >= 3:
            print('in aosTeleState:')
            print(self.stateV)
            print(self.opdGrid1d.shape)
            print(self.opdGrid1d[0])
            print(self.opdGrid1d[-1])
            print(self.opdGrid1d[-2])

        if hasattr(self, 'zAngle'):
            # M1M3 gravitational and thermal
            printthx = M1M3.zdx * \
                np.cos(self.zAngle) + M1M3.hdx * np.sin(self.zAngle)
            printthy = M1M3.zdy * \
                np.cos(self.zAngle) + M1M3.hdy * np.sin(self.zAngle)
            printthz = M1M3.zdz * \
                np.cos(self.zAngle) + M1M3.hdz * np.sin(self.zAngle)
            u0 = M1M3.zf * np.cos(self.zAngle) + M1M3.hf * np.sin(self.zAngle)

            # convert dz to grid sag
            x, y, _ = ct.ZCRS2M1CRS(M1M3.bx, M1M3.by, M1M3.bz)
            # M1M3.idealShape() uses mm everywhere
            zpRef = M1M3.idealShape((x + printthx) * 1000,
                                    (y + printthy) * 1000, M1M3.nodeID) / 1000
            zRef = M1M3.idealShape(x * 1000, y * 1000, M1M3.nodeID) / 1000
            printthz = printthz - (zpRef - zRef)
            zc = ZernikeAnnularFit(printthz, x / M1M3.R,
                                   y / M1M3.R, 3, M1M3.Ri / M1M3.R)
            printthz = printthz - ZernikeAnnularEval(
                zc, x / M1M3.R, y / M1M3.R, M1M3.Ri / M1M3.R)

            LUTforce = getLUTforce(self.zAngle / np.pi * 180, M1M3.LUTfile)
            # add 5% force error
            np.random.seed(self.iSim)
            # if the error is a percentage error
            # myu = (1+2*(np.random.rand(M1M3.nActuator)-0.5)
            #        *self.M1M3ForceError)*LUTforce
            # if the error is a absolute error in Newton
            myu = 2 * (np.random.rand(M1M3.nActuator) - 0.5) \
                    * self.M1M3ForceError + LUTforce
            # balance forces along z
            myu[M1M3.nzActuator - 1] = np.sum(LUTforce[:M1M3.nzActuator]) \
                - np.sum(myu[:M1M3.nzActuator - 1])
            # ; %balance forces along y
            myu[M1M3.nActuator - 1] = np.sum(LUTforce[M1M3.nzActuator:]) \
                - np.sum(myu[M1M3.nzActuator:-1])

            self.M1M3surf = (printthz + M1M3.G.dot(myu - u0)
                             ) * 1e6  # now in um

            # M2
            self.M2surf = M2.zdz * np.cos(self.zAngle) \
                + M2.hdz * np.sin(self.zAngle)
            pre_comp_elev = 0
            self.M2surf -= M2.zdz * np.cos(pre_comp_elev) \
                + M2.hdz * np.sin(pre_comp_elev)

        if hasattr(self, 'T'):

            self.M1M3surf += self.T * M1M3.tbdz + self.M1M3TxGrad * M1M3.txdz \
                + self.M1M3TyGrad * M1M3.tydz + self.M1M3TzGrad * M1M3.tzdz \
                + self.M1M3TrGrad * M1M3.trdz

            self.M2surf += self.M2TzGrad * M2.tzdz + self.M2TrGrad * M2.trdz

        if hasattr(self, 'M1M3surf'):
            _, _, self.M1M3surf = ct.M1CRS2ZCRS(x, y, self.M1M3surf)
        if hasattr(self, 'M2surf'):
            _, _, self.M2surf = ct.M2CRS2ZCRS(x, y, self.M2surf)

        if hasattr(self, 'camRot'):

            pre_elev = 0
            pre_camR = 0
            pre_temp_camR = 0
            self.getCamDistortion('L1RB', pre_elev, pre_camR, pre_temp_camR)
            self.getCamDistortion('L2RB', pre_elev, pre_camR, pre_temp_camR)
            self.getCamDistortion('FRB', pre_elev, pre_camR, pre_temp_camR)
            self.getCamDistortion('L3RB', pre_elev, pre_camR, pre_temp_camR)
            self.getCamDistortion('FPRB', pre_elev, pre_camR, pre_temp_camR)
            self.getCamDistortion('L1S1zer', pre_elev, pre_camR, pre_temp_camR)
            self.getCamDistortion('L2S1zer', pre_elev, pre_camR, pre_temp_camR)
            self.getCamDistortion('L3S1zer', pre_elev, pre_camR, pre_temp_camR)
            self.getCamDistortion('L1S2zer', pre_elev, pre_camR, pre_temp_camR)
            self.getCamDistortion('L2S2zer', pre_elev, pre_camR, pre_temp_camR)
            self.getCamDistortion('L3S2zer', pre_elev, pre_camR, pre_temp_camR)

    def getCamDistortion(self, distType, pre_elev, pre_camR, pre_temp_camR):
        dataFile = os.path.join('data/camera', (distType + '.txt'))
        data = np.loadtxt(dataFile, skiprows=1)
        distortion = data[0, 3:] * np.cos(self.zAngle) +\
            (data[1, 3:] * np.cos(self.camRot) +
             data[2, 3:] * np.sin(self.camRot)) * np.sin(self.zAngle)
        # pre-compensation
        distortion -= data[0, 3:] * np.cos(pre_elev) +\
            (data[1, 3:] * np.cos(pre_camR) +
             data[2, 3:] * np.sin(pre_camR)) * np.sin(pre_elev)

        # simple temperature interpolation/extrapolation
        if self.camTB < data[3, 2]:
            distortion += data[3, 3:]
        elif self.camTB > data[10, 2]:
            distortion += data[10, 3:]
        else:
            p2 = (data[3:, 2] > self.camTB).argmax() + 3
            p1 = p2 - 1
            w1 = (data[p2, 2] - self.camTB) / (data[p2, 2] - data[p1, 2])
            w2 = (self.camTB - data[p1, 2]) / (data[p2, 2] - data[p1, 2])
            distortion += w1 * data[p1, 3:] + w2 * data[p2, 3:]

        distortion -= data[(data[3:, 2] == pre_temp_camR).argmax() + 3, 3:]
        # Andy's Zernike order is different, fix it
        if distType[-3:] == 'zer':
            zidx = [1, 3, 2, 5, 4, 6, 8, 9, 7, 10, 13, 14, 12, 15, 11, 19,
                    18, 20, 17, 21, 16, 25, 24, 26, 23, 27, 22, 28]
            distortion = distortion[[x - 1 for x in zidx]]
        setattr(self, distType, distortion)

    def update(self, ctrl):
        self.stateV += ctrl.uk
        if np.any(self.stateV > ctrl.range):
            ii = (self.stateV > ctrl.range).argmax()
            raise RuntimeError("ERROR: stateV[%d] = %e > its range = %e" % (
                ii, self.stateV[ii], ctrl.range[ii]))

    def writePertFile(self, ndofA):
        fid = open(self.pertFile, 'w')
        for i in range(ndofA):
            if (self.stateV[i] != 0):
                # don't add comments after each move command,
                # Phosim merges all move commands into one!
                fid.write('move %d %7.4f \n' % (
                    self.phosimActuatorID[i], self.stateV[i]))
        fid.close()
        np.savetxt(self.pertMatFile, self.stateV)

    def setIterNo(self, metr, iIter, wfs=None):
        self.iIter = iIter
        #leave last digit for wavelength
        self.obsID = 9000000 + self.iSim * 1000 + self.iIter * 10
        self.pertFile = '%s/iter%d/sim%d_iter%d_pert.txt' % (
            self.pertDir, self.iIter, self.iSim, self.iIter)
        self.pertMatFile = '%s/iter%d/sim%d_iter%d_pert.mat' % (
            self.pertDir, self.iIter, self.iSim, self.iIter)
        if not os.path.exists('%s/iter%d/' % (self.imageDir, self.iIter)):
            os.makedirs('%s/iter%d/' % (self.imageDir, self.iIter))
        if not os.path.exists('%s/iter%d/' % (self.pertDir, self.iIter)):
            os.makedirs('%s/iter%d/' % (self.pertDir, self.iIter))

        metr.PSSNFile = '%s/iter%d/sim%d_iter%d_PSSN.txt' % (
            self.imageDir, self.iIter, self.iSim, self.iIter)
        metr.elliFile = '%s/iter%d/sim%d_iter%d_elli.txt' % (
            self.imageDir, self.iIter, self.iSim, self.iIter)
        if wfs is not None:
            wfs.zFile = '%s/iter%d/sim%d_iter%d.z4c' % (
                self.imageDir, self.iIter, self.iSim, self.iIter)
            wfs.catFile = '%s/iter%d/wfs_catalog.txt' % (
                self.pertDir, self.iIter)
            wfs.zCompFile = '%s/iter%d/checkZ4C_iter%d.png' % (
                self.pertDir, self.iIter, self.iIter)
        
        self.OPD_inst = []
        for irun in range(self.nOPDrun):
            if self.nOPDrun == 1:
                self.OPD_inst.append('%s/iter%d/sim%d_iter%d_opd%d.inst' % (
                    self.pertDir, self.iIter, self.iSim, self.iIter,
                    metr.nFieldp4))
            else:
                self.OPD_inst.append(
                    '%s/iter%d/sim%d_iter%d_opd%d_w%d.inst' % (
                    self.pertDir, self.iIter, self.iSim, self.iIter,
                    metr.nFieldp4, irun))
        self.OPD_cmd = '%s/iter%d/sim%d_iter%d_opd%d.cmd' % (
            self.pertDir, self.iIter, self.iSim, self.iIter, metr.nFieldp4)

        self.OPD_log = []
        for irun in range(self.nOPDrun):
            if self.nOPDrun == 1:
                self.OPD_log.append('%s/iter%d/sim%d_iter%d_opd%d.log' % (
                    self.imageDir, self.iIter, self.iSim, self.iIter,
                    metr.nFieldp4))
            else:
                self.OPD_log.append('%s/iter%d/sim%d_iter%d_opd%d_w%d.log' % (
                        self.imageDir, self.iIter, self.iSim, self.iIter,
                        metr.nFieldp4, irun))
                
        self.zTrueFile = []
        for irun in range(self.nOPDrun):
            if self.nOPDrun == 1:        
                self.zTrueFile.append('%s/iter%d/sim%d_iter%d_opd.zer' % (
                    self.imageDir, self.iIter, self.iSim, self.iIter))
            else:
                self.zTrueFile.append('%s/iter%d/sim%d_iter%d_opd_w%d.zer' % (
                    self.imageDir, self.iIter, self.iSim, self.iIter, irun))

        if iIter > 0:
            self.zTrueFile_m1 = []
            for irun in range(self.nOPDrun):
                if self.nOPDrun == 1:
                    self.zTrueFile_m1.append(
                        '%s/iter%d/sim%d_iter%d_opd.zer' % (
                        self.imageDir, self.iIter - 1, self.iSim,
                            self.iIter - 1))
                else:
                    self.zTrueFile_m1.append(
                        '%s/iter%d/sim%d_iter%d_opd_w%d.zer' % (
                        self.imageDir, self.iIter - 1, self.iSim,
                            self.iIter - 1, irun))
                
            self.pertMatFile_m1 = '%s/iter%d/sim%d_iter%d_pert.mat' % (
                self.pertDir, self.iIter - 1, self.iSim, self.iIter - 1)
            self.stateV = np.loadtxt(self.pertMatFile_m1)
            self.pertMatFile_0 = '%s/iter0/sim%d_iter0_pert.mat' % (
                self.pertDir, self.iSim)
            self.stateV0 = np.loadtxt(self.pertMatFile_0)
            if wfs is not None:
                wfs.zFile_m1 = '%s/iter%d/sim%d_iter%d.z4c' % (
                    self.imageDir, self.iIter - 1, self.iSim, self.iIter - 1)

            # PSSN from last iteration needs to be known for shiftGear
            if not (hasattr(metr, 'GQFWHMeff')):
                metr.PSSNFile_m1 = '%s/iter%d/sim%d_iter%d_PSSN.txt' % (
                    self.imageDir, self.iIter - 1, self.iSim, self.iIter - 1)
                aa = np.loadtxt(metr.PSSNFile_m1)
                metr.GQFWHMeff = aa[1, -1]

    def getOPDAll(self, opdoff, metr, numproc, znwcs,
                  obscuration, debugLevel):

        if not opdoff:
            self.writeOPDinst(metr)
            self.writeOPDcmd(metr)
            argList = []
            for i in range(self.nOPDrun):
                    
                srcFile = '%s/output/opd_%d.fits.gz' % (
                    self.phosimDir, self.obsID + i)
                dstFile = '%s/iter%d/sim%d_iter%d_opd_w%d.fits.gz' % (
                    self.imageDir, self.iIter, self.iSim, self.iIter, i)

                argList.append((self.OPD_inst[i], self.OPD_cmd, self.inst,
                                    self.eimage, self.OPD_log[i],
                                    self.phosimDir,
                                    self.zTrueFile[i], metr.nFieldp4,
                                    znwcs, obscuration, self.opdx, self.opdy,
                                    srcFile, dstFile, debugLevel))
                if sys.platform == 'darwin':
                    runOPD1w(argList[i])
                
            if sys.platform != 'darwin':
                # test, pdb cannot go into the subprocess
                # runOPD1w(argList[0])
                pool = multiprocessing.Pool(numproc)
                pool.map(runOPD1w, argList)
                pool.close()
                pool.join()
            
            
    def getOPDAllfromBase(self, baserun, metr):
        for i in range(self.nOPDrun):
            if not os.path.isfile(self.OPD_inst[i]):
                baseFile = self.OPD_inst[i].replace(
                    'sim%d' % self.iSim, 'sim%d' % baserun)
                os.link(baseFile, self.OPD_inst[i])

            if not os.path.isfile(self.OPD_log[i]):
                baseFile = self.OPD_log[i].replace(
                    'sim%d' % self.iSim, 'sim%d' % baserun)
                os.link(baseFile, self.OPD_log[i])

            for iField in range(metr.nFieldp4):
                opdFile = '%s/iter%d/sim%d_iter%d_opd%d_w%d.fits' % (
                    self.imageDir, self.iIter, self.iSim, self.iIter,
                    iField, i)
                if not os.path.isfile(opdFile):
                    baseFile = opdFile.replace(
                        'sim%d' % self.iSim, 'sim%d' % baserun)
                    os.link(baseFile, opdFile)
                
        if not os.path.isfile(self.OPD_cmd):
            baseFile = self.OPD_cmd.replace(
                'sim%d' % self.iSim, 'sim%d' % baserun)
            os.link(baseFile, self.OPD_cmd)

        if not os.path.isfile(self.zTrueFile):
            baseFile = self.zTrueFile.replace(
                'sim%d' % self.iSim, 'sim%d' % baserun)
            os.link(baseFile, self.zTrueFile)


    def writeOPDinst(self, metr):
        for irun in range(self.nOPDrun):
            fid = open(self.OPD_inst[irun], 'w')
            fid.write('Opsim_filter %d\n\
Opsim_obshistid %d\n\
SIM_VISTIME 15.0\n\
SIM_NSNAP 1\n' % (phosimFilterID[self.band], self.obsID + irun))
            fpert = open(self.pertFile, 'r')
            fid.write(fpert.read())
            for i in range(metr.nFieldp4):
                if self.nOPDrun == 1:
                    fid.write('opd %2d\t%9.6f\t%9.6f %5.1f\n' % (
                        i, metr.fieldX[i], metr.fieldY[i],
                        self.effwave * 1e3))
                else:
                    fid.write('opd %2d\t%9.6f\t%9.6f %5.1f\n' % (
                        i, metr.fieldX[i], metr.fieldY[i],
                        self.GQwave[self.band][irun] * 1e3))
            fid.close()
            fpert.close()

    def writeOPDcmd(self, metr):
        fid = open(self.OPD_cmd, 'w')
        fid.write('zenith_v 1000.0\n\
raydensity 0.0\n\
perturbationmode 1\n')
        fid.close()

    def getPSFAll(self, psfoff, metr, numproc, debugLevel, pixelum=10):

        if not psfoff:
            self.writePSFinst(metr)
            self.writePSFcmd(metr)
            self.PSF_log = '%s/iter%d/sim%d_iter%d_psf%d.log' % (
                self.imageDir, self.iIter, self.iSim, self.iIter, metr.nField)

            if pixelum == 10:
                instiq = self.inst
            elif pixelum == 0.2:
                instiq = self.inst + 'iq'
            myargs = '%s -c %s -i %s -p %d -e %d > %s' % (
                self.PSF_inst, self.PSF_cmd, instiq, numproc, self.eimage,
                self.PSF_log)
            if debugLevel >= 2:
                print('********Runnnig PHOSIM with following \
                parameters********')
                print('Check the log file below for progress')
                print('%s' % myargs)

            runProgram('python %s/phosim.py' %
                       self.phosimDir, argstring=myargs)
            plt.figure(figsize=(10, 10))
            for i in range(metr.nField):
                if pixelum == 10:
                    chipStr, px, py = self.fieldXY2Chip(
                        metr.fieldXp[i], metr.fieldYp[i], debugLevel)
                    # no need to be too big, 10um pixel
                    self.psfStampSize = 128
                elif pixelum == 0.2:
                    chipStr = 'F%02d' % i
                    px = 2000
                    py = 2000
                src = glob.glob('%s/output/*%d*%s*E000.fit*' % (
                    self.phosimDir, self.obsID, chipStr))
                if len(src) == 0:
                    raise RuntimeError(
                        "cannot find Phosim output: osbID=%d, chipStr = %s" % (
                            self.obsID, chipStr))
                elif 'gz' in src[0]:
                    # when .fits and .fits.gz both exist
                    # which appears first seems random
                    runProgram('gunzip -f %s' % src[0])
                elif 'gz' in src[-1]:
                    runProgram('gunzip -f %s' % src[-1])

                fitsfile = src[0].replace('.gz', '')
                IHDU = fits.open(fitsfile)
                chipImage = IHDU[0].data
                IHDU.close()
                os.rename(fitsfile, fitsfile.replace('.fits','psf.fits'))

                psf = chipImage[
                    py - self.psfStampSize * 2:py + self.psfStampSize * 2,
                    px - self.psfStampSize * 2:px + self.psfStampSize * 2]
                offsety = np.argwhere(psf == psf.max())[0][0] - \
                    self.psfStampSize * 2 + 1
                offsetx = np.argwhere(psf == psf.max())[0][1] - \
                    self.psfStampSize * 2 + 1
                psf = chipImage[
                    py - self.psfStampSize / 2 + offsety:
                    py + self.psfStampSize / 2 + offsety,
                    px - self.psfStampSize / 2 + offsetx:
                    px + self.psfStampSize / 2 + offsetx]
                if debugLevel >= 3:
                    print('px = %d, py = %d' % (px, py))
                    print('offsetx = %d, offsety = %d' % (offsetx, offsety))
                    print('passed %d' % i)

                if pixelum == 10:
                    displaySize = 20
                elif pixelum == 0.2:
                    displaySize = 100

                dst = '%s/iter%d/sim%d_iter%d_psf%d.fits' % (
                    self.imageDir, self.iIter, self.iSim, self.iIter, i)
                if os.path.isfile(dst):
                    os.remove(dst)
                hdu = fits.PrimaryHDU(psf)
                hdu.writeto(dst)

                if self.inst[:4] == 'lsst':
                    if i == 0:
                        pIdx = 1
                    else:
                        pIdx = i + metr.nArm
                    nRow = metr.nRing + 1
                    nCol = metr.nArm
                elif self.inst[:6] == 'comcam':
                    aa = [7, 4, 1, 8, 5, 2, 9, 6, 3]
                    pIdx = aa[i]
                    nRow = 3
                    nCol = 3

                plt.subplot(nRow, nCol, pIdx)
                plt.imshow(extractArray(psf, displaySize),
                           origin='lower', interpolation='none')
                plt.title('%d' % i)
                plt.axis('off')

            # plt.show()
            pngFile = '%s/iter%d/sim%d_iter%d_psf.png' % (
                self.imageDir, self.iIter, self.iSim, self.iIter)
            plt.savefig(pngFile, bbox_inches='tight')
            plt.close()

    def getPSFAllfromBase(self, baserun, metr):
        self.PSF_inst = '%s/iter%d/sim%d_iter%d_psf%d.inst' % (
            self.pertDir, self.iIter, self.iSim, self.iIter, metr.nField)
        if not os.path.isfile(self.PSF_inst):
            baseFile = self.PSF_inst.replace(
                'sim%d' % self.iSim, 'sim%d' % baserun)
            # PSF files are not crucial, it is ok if the baserun doesn't have
            # it
            if os.path.isfile(baseFile):
                os.link(baseFile, self.PSF_inst)
            else:
                return

        self.PSF_cmd = '%s/iter%d/sim%d_iter%d_psf%d.cmd' % (
            self.pertDir, self.iIter, self.iSim, self.iIter, metr.nField)
        if not os.path.isfile(self.PSF_cmd):
            baseFile = self.PSF_cmd.replace(
                'sim%d' % self.iSim, 'sim%d' % baserun)
            os.link(baseFile, self.PSF_cmd)

        self.PSF_log = '%s/iter%d/sim%d_iter%d_psf%d.log' % (
            self.imageDir, self.iIter, self.iSim, self.iIter, metr.nField)
        if not os.path.isfile(self.PSF_log):
            baseFile = self.PSF_log.replace(
                'sim%d' % self.iSim, 'sim%d' % baserun)
            os.link(baseFile, self.PSF_log)

        for i in range(metr.nField):
            psfFile = '%s/iter%d/sim%d_iter%d_psf%d.fits' % (
                self.imageDir, self.iIter, self.iSim, self.iIter, i)
            if not os.path.isfile(psfFile):
                baseFile = psfFile.replace(
                    'sim%d' % self.iSim, 'sim%d' % baserun)
                os.link(baseFile, psfFile)

        pngFile = '%s/iter%d/sim%d_iter%d_psf.png' % (
            self.imageDir, self.iIter, self.iSim, self.iIter)
        if not os.path.isfile(pngFile):
            baseFile = pngFile.replace('sim%d' % self.iSim, 'sim%d' % baserun)
            os.link(baseFile, pngFile)

    def writePSFinst(self, metr):
        self.PSF_inst = '%s/iter%d/sim%d_iter%d_psf%d.inst' % (
            self.pertDir, self.iIter, self.iSim, self.iIter, metr.nField)
        fid = open(self.PSF_inst, 'w')
        fid.write('Opsim_filter %d\n\
Opsim_obshistid %d\n\
SIM_VISTIME 15.0\n\
SIM_NSNAP 1\n\
SIM_SEED %d\n\
SIM_CAMCONFIG 1\n' % (phosimFilterID[self.band], self.obsID,
                      self.obsID % 10000 + 31))
        fpert = open(self.pertFile, 'r')

        fid.write(fpert.read())
        for i in range(metr.nField):
            fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/%s 0.0 0.0 0.0 0.0 0.0 0.0 star none  none\n' % (
                i, metr.fieldXp[i], metr.fieldYp[i], self.psfMag, self.sedfile))
        fid.close()
        fpert.close()

    def writePSFcmd(self, metr):
        self.PSF_cmd = '%s/iter%d/sim%d_iter%d_psf%d.cmd' % (
            self.pertDir, self.iIter, self.iSim, self.iIter, metr.nField)
        fid = open(self.PSF_cmd, 'w')
        fid.write('zenith_v 1000.0\n\
raydensity 0.0\n\
perturbationmode 1\n\
trackingmode 0\n\
cleartracking\n\
clearturbulence\n\
clearopacity\n\
atmosphericdispersion 0\n\
lascatprob 0.0\n\
contaminationmode 0\n\
coatingmode 0\n\
airrefraction 0\n\
diffractionmode 1\n\
straylight 0\n\
detectormode 0\n')
        fid.close()
# clearperturbations\n\

    def getWFSAll(self, wfs, metr, numproc, debugLevel):

        self.writeWFScmd(wfs, -1)
        for iRun in range(wfs.nRun):
            self.WFS_log = '%s/iter%d/sim%d_iter%d_wfs%d.log' % (
                self.imageDir, self.iIter, self.iSim, self.iIter, wfs.nWFS)
            if wfs.nRun == 1:
                self.writeWFSinst(wfs, metr, -1)
            else:
                self.writeWFSinst(wfs, metr, iRun)
                self.WFS_log = self.WFS_log.replace(
                    '.log', '_%s.log' % (wfs.halfChip[iRun]))

            myargs = '%s -c %s -i %s -p %d -e %d > %s' % (
                self.WFS_inst, self.WFS_cmd, self.inst, numproc, self.eimage,
                self.WFS_log)
            if debugLevel >= 2:
                print('********Runnnig PHOSIM with following parameters\
                ********')
                print('Check the log file below for progress')
                print('%s' % myargs)

            runProgram('python %s/phosim.py' %
                       self.phosimDir, argstring=myargs)
            plt.figure(figsize=(10, 5))
            for i in range(metr.nFieldp4 - wfs.nWFS, metr.nFieldp4):
                chipStr, px, py = self.fieldXY2Chip(
                    metr.fieldXp[i], metr.fieldYp[i], debugLevel)
                src = glob.glob('%s/output/*%s*%s*E000.fit*' %
                                (self.phosimDir, self.obsID, chipStr))
                if wfs.nRun == 1:
                    for ioffset in [0, 1]:
                        if '.gz' in src[0]:
                            runProgram('gunzip -f %s' % src[ioffset])
                        chipFile = src[ioffset].replace('.gz', '')
                        runProgram('mv -f %s %s/iter%d' %
                                   (chipFile, self.imageDir, self.iIter))
                else:
                    if '.gz' in src[0]:
                        runProgram('gunzip -f %s' % src[0])
                    chipFile = src[0].replace('.gz', '')
                    targetFile = os.path.split(chipFile.replace(
                        'E000', '%s_E000' % wfs.halfChip[iRun]))[1]
                    runProgram('mv -f %s %s/iter%d/%s' %
                               (chipFile, self.imageDir, self.iIter,
                                targetFile))

    def writeWFSinst(self, wfs, metr, iRun=-1):
        # iRun = -1 means only need to run it once
        self.WFS_inst = '%s/iter%d/sim%d_iter%d_wfs%d.inst' % (
            self.pertDir, self.iIter, self.iSim, self.iIter, wfs.nWFS)
        if iRun != -1:
            self.WFS_inst = self.WFS_inst.replace(
                '.inst', '_%s.inst' % wfs.halfChip[iRun])

        fid = open(self.WFS_inst, 'w')
        fpert = open(self.pertFile, 'r')
        hasCamPiston = False
        for line in fpert:
            if iRun != -1 and line.split()[:2] == ['move', '10']:
                # move command follow Zemax coordinate system.
                fid.write('move 10 %9.4f\n' %
                          (float(line.split()[2]) - wfs.offset[iRun] * 1e3))
                hasCamPiston = True
            else:
                fid.write(line)
        if iRun != -1 and (not hasCamPiston):
            fid.write('move 10 %9.4f\n' % (-wfs.offset[iRun] * 1e3))

        fpert.close()
        fid.write('Opsim_filter %d\n\
Opsim_obshistid %d\n\
SIM_VISTIME 15.0\n\
SIM_NSNAP 1\n\
SIM_SEED %d\n\
Opsim_rawseeing 0.7283\n' % (phosimFilterID[self.band],
                             self.obsID, self.obsID % 10000 + 4))
        if self.inst[:4] == 'lsst':
            fid.write('SIM_CAMCONFIG 2\n')
        elif self.inst[:6] == 'comcam':
            fid.write('SIM_CAMCONFIG 1\n')

        ii = 0
        for i in range(metr.nFieldp4 - wfs.nWFS, metr.nFieldp4):
            if self.inst[:4] == 'lsst':
                if i % 2 == 1:  # field 31, 33, R44 and R00
                    fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/%s 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                        ii, metr.fieldXp[i] + 0.020, metr.fieldYp[i],
                        self.cwfsMag, self.sedfile))
                    ii += 1
                    fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/%s 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                        ii, metr.fieldXp[i] - 0.020, metr.fieldYp[i],
                        self.cwfsMag, self.sedfile))
                    ii += 1
                else:
                    fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/%s 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                        ii, metr.fieldXp[i], metr.fieldYp[i] + 0.020,
                        self.cwfsMag, self.sedfile))
                    ii += 1
                    fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/%s 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                        ii, metr.fieldXp[i], metr.fieldYp[i] - 0.020,
                        self.cwfsMag, self.sedfile))
                    ii += 1
            elif self.inst[:6] == 'comcam':
                fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/%s 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                    ii, metr.fieldXp[i], metr.fieldYp[i],
                    self.cwfsMag, self.sedfile))
                ii += 1
        fid.close()
        fpert.close()

    def writeWFScmd(self, wfs, iRun=-1):
        # iRun = -1 means only need to run it once
        self.WFS_cmd = '%s/iter%d/sim%d_iter%d_wfs%d.cmd' % (
            self.pertDir, self.iIter, self.iSim, self.iIter, wfs.nWFS)
        if iRun != -1:
            self.WFS_cmd = self.WFS_cmd.replace(
                '.cmd', '_%s.cmd' % wfs.halfChip[iRun])

        fid = open(self.WFS_cmd, 'w')
        fid.write('zenith_v 1000.0\n\
raydensity 0.0\n\
perturbationmode 1\n\
trackingmode 0\n\
cleartracking\n\
clearclouds\n\
lascatprob 0.0\n\
contaminationmode 0\n\
coatingmode 0\n\
diffractionmode 1\n\
straylight 0\n\
detectormode 0\n')
# airrefraction 0\n\

        # body command interferes with move commands;
        # let's not piston the detector only.
        # if iRun != -1:
        #     fid.write('body 11 5 %+4.1f\n'%(wfs.offset[iRun]))
        fid.close()

    def fieldXY2Chip(self, fieldX, fieldY, debugLevel):
        ruler = getChipBoundary(
            '%s/data/lsst/focalplanelayout.txt' % (self.phosimDir))

        # r for raft, c for chip, p for pixel
        rx, cx, px = fieldAgainstRuler(ruler, fieldX, 4000)
        ry, cy, py = fieldAgainstRuler(ruler, fieldY, 4072)

        if debugLevel >= 3:
            print('ruler:\n')
            print(ruler)
            print(len(ruler))

        return 'R%d%d_S%d%d' % (rx, ry, cx, cy), px, py


def runProgram(command, binDir=None, argstring=None):
    myCommand = command
    if binDir is not None:
        myCommand = os.path.join(binDir, command)
    if argstring is not None:
        myCommand += (' ' + argstring)
    if subprocess.call(myCommand, shell=True) != 0:
        raise RuntimeError("Error running %s" % myCommand)


def getChipBoundary(fplayoutFile):

    mydict = dict()
    f = open(fplayoutFile)
    for line in f:
        line = line.strip()
        if (line.startswith('R')):
            mydict[line.split()[0]] = [float(line.split()[1]),
                                       float(line.split()[2])]

    f.close()
    ruler = sorted(set([x[0] for x in mydict.values()]))
    return ruler


def fieldAgainstRuler(ruler, field, chipPixel):

    field = field * 180000  # degree to micron
    p2 = (ruler >= field)
    if (np.count_nonzero(p2) == 0):  # too large to be in range
        p = len(ruler) - 1  # p starts from 0
    elif (p2[0]):
        p = 0
    else:
        p1 = p2.argmax() - 1
        p2 = p2.argmax()
        if (ruler[p2] - field) < (field - ruler[p1]):
            p = p2
        else:
            p = p1

    pixel = (field - ruler[p]) / 10  # 10 for 10micron pixels
    pixel += chipPixel / 2

    return np.floor(p / 3), p % 3, pixel


def getLUTforce(zangle, LUTfile):
    """
    zangle should be in degree
    """

    lut = np.loadtxt(LUTfile)
    ruler = lut[0, :]

    step = ruler[1] - ruler[0]

    p2 = (ruler >= zangle)
#    print "FINE",p2, p2.shape
    if (np.count_nonzero(p2) == 0):  # zangle is too large to be in range
        p2 = ruler.shape[0] - 1
        p1 = p2
        w1 = 1
        w2 = 0
    elif (p2[0]):  # zangle is too small to be in range
        p2 = 0  # this is going to be used as index
        p1 = 0  # this is going to be used as index
        w1 = 1
        w2 = 0
    else:
        p1 = p2.argmax() - 1
        p2 = p2.argmax()
        w1 = (ruler[p2] - zangle) / step
        w2 = (zangle - ruler[p1]) / step

    return np.dot(w1, lut[1:, p1]) + np.dot(w2, lut[1:, p2])

def runOPD1w(argList):
    OPD_inst = argList[0]
    OPD_cmd = argList[1]
    inst = argList[2]
    eimage = argList[3]
    OPD_log = argList[4]
    phosimDir = argList[5]
    zTrueFile = argList[6]
    nFieldp4 = argList[7]
    znwcs = argList[8]
    obscuration = argList[9]
    opdx = argList[10]
    opdy = argList[11]
    srcFile = argList[12]
    dstFile = argList[13]
    debugLevel = argList[14]
    
    if debugLevel >= 3:
        runProgram('head %s' % OPD_inst)
        runProgram('head %s' % OPD_cmd)

    myargs = '%s -c %s -i %s -e %d > %s' % (
        OPD_inst, OPD_cmd, inst, eimage,
        OPD_log)
    if debugLevel >= 2:
        print('*******Runnnig PHOSIM with following parameters*******')
        print('Check the log file below for progress')
        print('%s' % myargs)
    runProgram('python %s/phosim.py' %
                   phosimDir, argstring=myargs)
    if debugLevel >= 3:
        print('DONE RUNNING PHOSIM FOR OPD: %s' % OPD_inst)
    if os.path.isfile(zTrueFile):
        os.remove(zTrueFile)
    fz = open(zTrueFile, 'ab')
    for i in range(nFieldp4):
        src = srcFile.replace('.fits.gz', '_%d.fits.gz' % i)
        dst = dstFile.replace('opd', 'opd%d' % i)
        shutil.move(src, dst)
        runProgram('gunzip -f %s' % dst)
        opdFile = dst.replace('.gz', '')
        IHDU = fits.open(opdFile)
        opd = IHDU[0].data  # Phosim OPD unit: um
        IHDU.close()
        idx = (opd != 0)
        Z = ZernikeAnnularFit(opd[idx], opdx[idx], opdy[idx],
                                  znwcs, obscuration)
        np.savetxt(fz, Z.reshape(1, -1), delimiter=' ')

    fz.close()

    if debugLevel >= 3:
        print(opdx)
        print(opdy)
        print(znwcs)
        print(obscuration)
    
