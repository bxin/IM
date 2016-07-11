#!/usr/bin/env python

# @author: Bo Xin
# @      Large Synoptic Survey Telescope

import os
import shutil
import glob
import subprocess

import numpy as np
from astropy.io import fits

from cwfsTools import ZernikeAnnularFit
from cwfsTools import extractArray

import matplotlib.pyplot as plt


class aosTeleState(object):

    def __init__(self, esti, instruFile, iSim, phosimDir,
                 pertDir, imageDir, debugLevel):
        # plan to write these to txt files. no columns for iter
        self.stateV = np.zeros(esti.ndofA)  # *np.nan # telescope state(?)

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
                    # by default we want micron for everything
                    if line.split()[3] == 'mm':
                        self.stateV[int(line.split()[1]) - 1] *= 1e3
                elif (line.startswith('opd_size')):
                    self.opdSize = int(line.split()[1])
                    if self.opdSize % 2 == 0:
                        self.opdSize -= 1
                elif (line.startswith('eimage')):
                    self.eimage = bool(int(line.split()[1]))
                elif (line.startswith('psf_mag')):
                    self.psfMag = int(line.split()[1])
                elif (line.startswith('psf_stamp_size')):
                    self.psfStampSize = int(line.split()[1])
                elif (line.startswith('cwfs_mag')):
                    self.cwfsMag = int(line.split()[1])
                elif (line.startswith('cwfs_stamp_size')):
                    self.cwfsStampSize = int(line.split()[1])

        fid.close()

        self.iSim = iSim
        self.phosimDir = phosimDir
        self.pertDir = pertDir
        self.imageDir = imageDir
        # self.setIterNo(0)
        self.phosimActuatorID = [
            # M2 z, x, y, rx, ry
            5, 6, 7, 8, 9,
            # Cam z, x, y, rx, ry
            10, 11, 12, 13, 14] + [
            # M13 and M2 bending
            i for i in range(15, 15 + esti.ndofA - 10)]
        self.wfsName = ['intra', 'extra']

        self.opdGrid1d = np.linspace(-1, 1, self.opdSize)
        self.opdx, self.opdy = np.meshgrid(self.opdGrid1d, self.opdGrid1d)
        runProgram('rm -rf %s/output/*'%self.phosimDir)
        
        if debugLevel >= 3:
            print('in aosTeleState:')
            print(self.stateV)

    def update(self, ctrl):
        self.stateV += ctrl.uk

    def writePertFile(self, esti):
        fid = open(self.pertFile, 'w')
        for i in range(esti.ndofA):
            if (self.stateV[i] != 0):
                #don't add comments after each move command,
                #Phosim merges all move commands into one!
                fid.write('move %d %7.4f \n' % ( 
                    self.phosimActuatorID[i], self.stateV[i], i + 1))
        fid.close()
        np.savetxt(self.pertMatFile, self.stateV)

    def setIterNo(self, metr, iIter):
        self.iIter = iIter
        self.obsID = 9000000 + self.iSim*100 + self.iIter
        self.zFile = '%s/iter%d/sim%d_iter%d_opd.zer' % (
            self.imageDir, self.iIter, self.iSim, self.iIter)
        self.zFile_m1 = '%s/iter%d/sim%d_iter%d_opd.zer' % (
            self.imageDir, self.iIter - 1, self.iSim, self.iIter - 1)
        self.pertFile = '%s/iter%d/sim%d_iter%d_pert.txt' % (
            self.pertDir, self.iIter, self.iSim, self.iIter)
        self.pertMatFile = '%s/iter%d/sim%d_iter%d_pert.mat' % (
            self.pertDir, self.iIter, self.iSim, self.iIter)
        if not os.path.exists('%s/iter%d/'%(self.imageDir, self.iIter)):
            os.makedirs('%s/iter%d/'%(self.imageDir, self.iIter))
        if not os.path.exists('%s/iter%d/'%(self.pertDir, self.iIter)):
            os.makedirs('%s/iter%d/'%(self.pertDir, self.iIter))

        metr.PSSNFile = '%s/iter%d/sim%d_iter%d_PSSN.txt'%(
            self.imageDir, self.iIter, self.iSim, self.iIter)
        metr.elliFile = '%s/iter%d/sim%d_iter%d_elli.txt'%(
            self.imageDir, self.iIter, self.iSim, self.iIter)
            
    def getOPD35(self, wfs, metr, numproc, wavelength, debugLevel):

        self.writeOPDinst(metr, wavelength)
        self.writeOPDcmd()
        self.OPD_log = '%s/iter%d/sim%d_iter%d_opd35.log' % (
            self.imageDir, self.iIter, self.iSim, self.iIter)

        if debugLevel >= 3:
            runProgram('head %s' % self.OPD_inst)
            runProgram('head %s' % self.OPD_cmd)

        myargs = '%s -c %s -p %d -e %d > %s' % (
            self.OPD_inst, self.OPD_cmd, numproc, self.eimage, self.OPD_log)
        if debugLevel >= 3:
            print('*******Runnnig PHOSIM with following parameters*******')
            print('Check the log file below for progress')
            print('%s' % myargs)
        runProgram('python %s/phosim.py' % self.phosimDir, argstring=myargs)
        if debugLevel >= 3:
            print('DONE RUNNING PHOSIM FOR OPD')
        if os.path.isfile(self.zFile):
            os.remove(self.zFile)
        fz = open(self.zFile, 'ab')
        for i in range(metr.nFieldp4):
            src = '%s/output/opd_%d_%d.fits.gz' % (
                self.phosimDir, self.obsID, i)
            dst = '%s/iter%d/sim%d_iter%d_opd%d.fits.gz' % (
                self.imageDir, self.iIter, self.iSim, self.iIter, i)
            shutil.move(src, dst)
            runProgram('gunzip -f %s' % dst)
            opdFile = dst.replace('.gz', '')
            IHDU = fits.open(opdFile)
            opd = IHDU[0].data  # Phosim OPD unit: um
            IHDU.close()
            idx = (opd != 0)
            Z = ZernikeAnnularFit(opd[idx], self.opdx[idx], self.opdy[idx],
                                  wfs.znwcs, wfs.inst.obscuration)
            np.savetxt(fz, Z.reshape(1, -1), delimiter=' ')

        fz.close()

        if debugLevel >= 3:
            print(self.opdGrid1d.shape)
            print(self.opdGrid1d[0])
            print(self.opdGrid1d[-1])
            print(self.opdGrid1d[-2])
            print(self.opdx)
            print(self.opdy)
            print(wfs.znwcs)
            print(wfs.inst.obscuration)

    def getOPD35fromBase(self, baserun, metr):
        self.OPD_inst = '%s/iter%d/sim%d_iter%d_opd35.inst' % (
            self.pertDir, self.iIter, self.iSim, self.iIter)
        if not os.path.isfile(self.OPD_inst):
            baseFile = self.OPD_inst.replace('sim%d'%self.iSim, 'sim%d'%baserun)
            os.link(baseFile, self.OPD_inst)

        self.OPD_cmd = '%s/iter%d/sim%d_iter%d_opd35.cmd' % (
            self.pertDir, self.iIter, self.iSim, self.iIter)
        if not os.path.isfile(self.OPD_cmd):
            baseFile = self.OPD_cmd.replace('sim%d'%self.iSim, 'sim%d'%baserun)
            os.link(baseFile, self.OPD_cmd)
        
        self.OPD_log = '%s/iter%d/sim%d_iter%d_opd35.log' % (
            self.imageDir, self.iIter, self.iSim, self.iIter)
        if not os.path.isfile(self.OPD_log):
            baseFile = self.OPD_log.replace('sim%d'%self.iSim, 'sim%d'%baserun)
            os.link(baseFile, self.OPD_log)

        if not os.path.isfile(self.zFile):
            baseFile = self.zFile.replace('sim%d'%self.iSim, 'sim%d'%baserun)
            os.link(baseFile, self.zFile)
        
        for i in range(metr.nFieldp4):
            opdFile = '%s/iter%d/sim%d_iter%d_opd%d.fits' % (
                self.imageDir, self.iIter, self.iSim, self.iIter, i)
            if not os.path.isfile(opdFile):
                baseFile = opdFile.replace('sim%d'%self.iSim, 'sim%d'%baserun)
                os.link(baseFile, opdFile)
                                                                    
    def writeOPDinst(self, metr, wavelength):
        self.OPD_inst = '%s/iter%d/sim%d_iter%d_opd35.inst' % (
            self.pertDir, self.iIter, self.iSim, self.iIter)
        fid = open(self.OPD_inst, 'w')
        fid.write('Opsim_filter 1\n\
Opsim_obshistid %d\n\
SIM_VISTIME 15.0\n\
SIM_NSNAP 1\n'%(self.obsID))
        fpert = open(self.pertFile, 'r')
        fid.write(fpert.read())
        for i in range(metr.nFieldp4):
            fid.write('opd %2d\t%9.6f\t%9.6f %5.1f\n' % (
                i, metr.fieldX[i], metr.fieldY[i], wavelength * 1e3))
        fid.close()
        fpert.close()

    def writeOPDcmd(self):
        self.OPD_cmd = '%s/iter%d/sim%d_iter%d_opd35.cmd' % (
            self.pertDir, self.iIter, self.iSim, self.iIter)
        fid = open(self.OPD_cmd, 'w')
        fid.write('zenith_v 1000.0\n\
raydensity 0.0\n\
perturbationmode 1\n')
        fid.close()

    def getPSF31(self, metr, numproc, debugLevel):

        self.writePSFinst(metr)
        self.writePSFcmd()
        self.PSF_log = '%s/iter%d/sim%d_iter%d_psf31.log' % (
            self.imageDir, self.iIter, self.iSim, self.iIter)

        myargs = '%s -c %s -p %d -e %d > %s' % (
            self.PSF_inst, self.PSF_cmd, numproc, self.eimage, self.PSF_log)
        if debugLevel >= 3:
            print('********Runnnig PHOSIM with following parameters********')
            print('Check the log file below for progress')
            print('%s' % myargs)

        runProgram('python %s/phosim.py' % self.phosimDir, argstring=myargs)
        plt.figure(figsize=(10, 10))
        for i in range(metr.nField):
            chipStr, px, py = self.fieldXY2Chip(
                metr.fieldXp[i], metr.fieldYp[i], debugLevel)
            src = glob.glob('%s/output/*%d*%s*' % (
                self.phosimDir, self.obsID, chipStr))
            if 'gz' in src[0]:
                runProgram('gunzip -f %s' % src[0])
            IHDU = fits.open(src[0].replace('.gz', ''))
            chipImage = IHDU[0].data
            IHDU.close()
            psf = chipImage[
                py - self.psfStampSize / 2:py + self.psfStampSize / 2,
                px - self.psfStampSize / 2:px + self.psfStampSize / 2]
            offsety = np.argwhere(psf == psf.max())[0][0] - \
                self.psfStampSize / 2 + 1
            offsetx = np.argwhere(psf == psf.max())[0][1] - \
                self.psfStampSize / 2 + 1
            psf = chipImage[
                py - self.psfStampSize / 2 + offsety:
                py + self.psfStampSize / 2 + offsety,
                px - self.psfStampSize / 2 + offsetx:
                px + self.psfStampSize / 2 + offsetx]

            if i == 0:
                pIdx = 1
            else:
                pIdx = i + metr.nArm

            dst = '%s/iter%d/sim%d_iter%d_psf%d.fits' % (
                self.imageDir, self.iIter, self.iSim, self.iIter, i)
            if os.path.isfile(dst):
                os.remove(dst)
            hdu = fits.PrimaryHDU(psf)
            hdu.writeto(dst)

            plt.subplot(metr.nRing + 1, metr.nArm, pIdx)
            plt.imshow(extractArray(psf, 20), origin='lower')
            plt.title('%d' % i)
            plt.axis('off')

            if debugLevel >= 3:
                print('px = %d, py = %d' % (px, py))
                print('offsetx = %d, offsety = %d' % (offsetx, offsety))
                print('passed %d' % i)

        # plt.show()
        pngFile = '%s/iter%d/sim%d_iter%d_psf.png' % (
            self.imageDir, self.iIter, self.iSim, self.iIter)
        plt.savefig(pngFile, bbox_inches='tight')

    def getPSF31fromBase(self, baserun, metr):
        self.PSF_inst = '%s/iter%d/sim%d_iter%d_psf31.inst' % (
            self.pertDir, self.iIter, self.iSim, self.iIter)
        if not os.path.isfile(self.PSF_inst):
            baseFile = self.PSF_inst.replace('sim%d'%self.iSim, 'sim%d'%baserun)
            os.link(baseFile, self.PSF_inst)

        self.PSF_cmd = '%s/iter%d/sim%d_iter%d_psf31.cmd' % (
            self.pertDir, self.iIter, self.iSim, self.iIter)
        if not os.path.isfile(self.PSF_cmd):
            baseFile = self.PSF_cmd.replace('sim%d'%self.iSim, 'sim%d'%baserun)
            os.link(baseFile, self.PSF_cmd)

        self.PSF_log = '%s/iter%d/sim%d_iter%d_psf31.log' % (
            self.imageDir, self.iIter, self.iSim, self.iIter)
        if not os.path.isfile(self.PSF_log):
            baseFile = self.PSF_log.replace('sim%d'%self.iSim, 'sim%d'%baserun)
            os.link(baseFile, self.PSF_log)
        
        for i in range(metr.nField):
            psfFile = '%s/iter%d/sim%d_iter%d_psf%d.fits' % (
                self.imageDir, self.iIter, self.iSim, self.iIter, i)
            if not os.path.isfile(psfFile):
                baseFile = psfFile.replace('sim%d'%self.iSim, 'sim%d'%baserun)
                os.link(baseFile, psfFile)

        pngFile = '%s/iter%d/sim%d_iter%d_psf.png' % (
            self.imageDir, self.iIter, self.iSim, self.iIter)
        if not os.path.isfile(pngFile):
            baseFile = pngFile.replace('sim%d'%self.iSim, 'sim%d'%baserun)
            os.link(baseFile, pngFile)
                                                                              
    def writePSFinst(self, metr):
        self.PSF_inst = '%s/iter%d/sim%d_iter%d_psf31.inst' % (
            self.pertDir, self.iIter, self.iSim, self.iIter)
        fid = open(self.PSF_inst, 'w')
        fid.write('Opsim_filter 1\n\
Opsim_obshistid %d\n\
SIM_VISTIME 15.0\n\
SIM_NSNAP 1\n'%(self.obsID))
        fpert = open(self.pertFile, 'r')
        fid.write(fpert.read())
        for i in range(metr.nField):
            fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/sed_500.txt 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                i, metr.fieldXp[i], metr.fieldYp[i], self.psfMag))
        fid.close()
        fpert.close()

    def writePSFcmd(self):
        self.PSF_cmd = '%s/iter%d/sim%d_iter%d_psf31.cmd' % (
            self.pertDir, self.iIter, self.iSim, self.iIter)
        fid = open(self.PSF_cmd, 'w')
        fid.write('zenith_v 1000.0\n\
raydensity 0.0\n\
perturbationmode 1\n')
        fid.close()

    def getWFS4(self, wfs, metr, numproc, debugLevel):

        self.writeWFSinst(metr)
        self.writeWFScmd()
        self.WFS_log = '%s/iter%d/sim%d_iter%d_wfs4.log' % (
            self.imageDir, self.iIter, self.iSim, self.iIter)

        myargs = '%s -c %s -p %d -e %d > %s' % (
            self.WFS_inst, self.WFS_cmd, numproc, self.eimage, self.WFS_log)
        if debugLevel >= 3:
            print('********Runnnig PHOSIM with following parameters********')
            print('Check the log file below for progress')
            print('%s' % myargs)

        runProgram('python %s/phosim.py' % self.phosimDir, argstring=myargs)
        plt.figure(figsize=(10, 5))
        for i in range(metr.nField, metr.nFieldp4):
            chipStr, px, py = self.fieldXY2Chip(
                metr.fieldXp[i], metr.fieldYp[i], debugLevel)
            src = glob.glob('%s/output/*%s*.gz' % (self.phosimDir, chipStr))
            for ioffset in [0, 1]:
                runProgram('gunzip -f %s' % src[ioffset])
                chipFile = src[ioffset].replace('.gz', '')
                shutil.move(chipFile, '%s/iter%d'%(self.imageDir, self.iIter))

    def writeWFSinst(self, metr):
        self.WFS_inst = '%s/iter%d/sim%d_iter%d_wfs4.inst' % (
            self.pertDir, self.iIter, self.iSim, self.iIter)
        fid = open(self.WFS_inst, 'w')
        fpert = open(self.pertFile, 'r')
        fid.write(fpert.read())
        fid.write('Opsim_filter 1\n\
Opsim_obshistid %d\n\
SIM_VISTIME 15.0\n\
SIM_NSNAP 1\n\
SIM_CAMCONFIG 7\n' % (self.obsID))
        ii = 0
        for i in range(metr.nField, metr.nFieldp4):
            if i % 2 == 1:  # field 31, 33, R44 and R00
                fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/sed_500.txt 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                    ii, metr.fieldXp[i] + 0.008, metr.fieldYp[i],
                    self.cwfsMag))
                ii += 1
                fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/sed_500.txt 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                    ii, metr.fieldXp[i] - 0.008, metr.fieldYp[i],
                    self.cwfsMag))
                ii += 1
            else:
                fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/sed_500.txt 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                    ii, metr.fieldXp[i], metr.fieldYp[i] + 0.008,
                    self.cwfsMag))
                ii += 1
                fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/sed_500.txt 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                    ii, metr.fieldXp[i], metr.fieldYp[i] - 0.008,
                    self.cwfsMag))
                ii += 1
        fid.close()
        fpert.close()

    def writeWFScmd(self):
        self.WFS_cmd = '%s/iter%d/sim%d_iter%d_wfs4.cmd' % (
            self.pertDir, self.iIter, self.iSim, self.iIter)
        fid = open(self.WFS_cmd, 'w')
        fid.write('zenith_v 1000.0\n\
raydensity 0.0\n\
opticsonlymode 1\n\
perturbationmode 1\n')
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
