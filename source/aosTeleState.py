#!/usr/bin/env python

# @author: Bo Xin
# @      Large Synoptic Survey Telescope

import os
import shutil
import glob
import subprocess

import numpy as np
from astropy.io import fits
import aosCoTransform as ct

from lsst.cwfs.tools import ZernikeAnnularFit
from lsst.cwfs.tools import ZernikeAnnularEval
from lsst.cwfs.tools import extractArray

import matplotlib.pyplot as plt


class aosTeleState(object):

    def __init__(self, inst, esti, M1M3, M2, instruFile, iSim, phosimDir,
                 pertDir, imageDir, debugLevel):
        # plan to write these to txt files. no columns for iter
        self.stateV = np.zeros(esti.ndofA)  # *np.nan # telescope state(?)

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
                elif (line.startswith('budget')): #read in in mas, convert to arcsec
                    self.budget = np.sqrt(np.sum([float(x)**2 for x in line.split()[1:]]))*1e-3
                elif (line.startswith('zenithAngle')):
                    self.zAngle = float(line.split()[1])/180*np.pi
                elif (line.startswith('temperature')):
                    self.T = float(line.split()[1])
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

        self.opdGrid1d = np.linspace(-1, 1, self.opdSize)
        self.opdx, self.opdy = np.meshgrid(self.opdGrid1d, self.opdGrid1d)
        # runProgram('rm -rf %s/output/*'%self.phosimDir)

        if debugLevel >= 3:
            print('in aosTeleState:')
            print(self.stateV)

        if hasattr(self, 'zAngle'):
            # M1M3 gravitational and thermal
            printthx = M1M3.zdx * np.cos(self.zAngle) + M1M3.hdx * np.sin(self.zAngle)
            printthy = M1M3.zdy * np.cos(self.zAngle) + M1M3.hdy * np.sin(self.zAngle)
            printthz = M1M3.zdz * np.cos(self.zAngle) + M1M3.hdz * np.sin(self.zAngle)
            u0 = M1M3.zf * np.cos(self.zAngle) + M1M3.hf * np.sin(self.zAngle)

            # convert dz to grid sag
            x, y, _ = ct.ZCRS2M1CRS(M1M3.bx, M1M3.by, M1M3.bz)
            #M1M3.idealShape() uses mm everywhere
            zpRef = M1M3.idealShape( (x+printthx)*1000,
                                    (y+printthy)*1000, M1M3.nodeID)/1000
            zRef = M1M3.idealShape(x*1000, y*1000, M1M3.nodeID)/1000
            printthz = printthz-(zpRef-zRef)
            zc = ZernikeAnnularFit(printthz, x/M1M3.R, y/M1M3.R, 3, M1M3.Ri/M1M3.R)
            printthz = printthz - ZernikeAnnularEval(
                zc, x/M1M3.R, y/M1M3.R, M1M3.Ri/M1M3.R)

            LUTforce = getLUTforce(self.zAngle/np.pi*180, M1M3.LUTfile)
            # add 5% force error
            np.random.seed(self.iSim)
            # if the error is a percentage error
            # myu = (1+2*(np.random.rand(M1M3.nActuator)-0.5)
            #        *self.M1M3ForceError)*LUTforce
            # if the error is a absolute error in Newton
            myu = 2*(np.random.rand(M1M3.nActuator)-0.5) \
                    *self.M1M3ForceError + LUTforce 
            #; %balance forces along z
            myu[M1M3.nzActuator-1]=np.sum(LUTforce[:M1M3.nzActuator]) \
                -np.sum(myu[:M1M3.nzActuator-1])
            # ; %balance forces along y
            myu[M1M3.nActuator-1]=np.sum(LUTforce[M1M3.nzActuator:]) \
              -np.sum(myu[M1M3.nzActuator:-1])
            
            self.M1M3surf = (printthz + M1M3.G.dot(myu - u0))*1e6 #now in um

            # M2
            self.M2surf = M2.zdz * np.cos(self.zAngle) + M2.hdz * np.sin(self.zAngle)
            
        if hasattr(self, 'T'):
            
            self.M1M3surf += self.T * M1M3.tbdz + self.M1M3TxGrad * M1M3.txdz \
              + self.M1M3TyGrad * M1M3.tydz + self.M1M3TzGrad * M1M3.tzdz \
              + self.M1M3TrGrad * M1M3.trdz

            _, _ , self.M1M3surf = ct.M1CRS2ZCRS(x, y, self.M1M3surf )

            self.M2surf += self.M2TzGrad * M2.tzdz \
              + self.M2TrGrad * M2.trdz

            _, _ , self.M2surf = ct.M2CRS2ZCRS(x, y, self.M2surf )
              
    def update(self, ctrl):
        self.stateV += ctrl.uk

    def writePertFile(self, esti):
        fid = open(self.pertFile, 'w')
        for i in range(esti.ndofA):
            if (self.stateV[i] != 0):
                #don't add comments after each move command,
                #Phosim merges all move commands into one!
                fid.write('move %d %7.4f \n' % ( 
                    self.phosimActuatorID[i], self.stateV[i]))
        fid.close()
        np.savetxt(self.pertMatFile, self.stateV)

    def setIterNo(self, wfs, metr, iIter):
        self.iIter = iIter
        self.obsID = 9000000 + self.iSim*100 + self.iIter
        self.zTrueFile = '%s/iter%d/sim%d_iter%d_opd.zer' % (
            self.imageDir, self.iIter, self.iSim, self.iIter)
        wfs.zFile = '%s/iter%d/sim%d_iter%d.z4c' % (
            self.imageDir, self.iIter, self.iSim, self.iIter)
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
        wfs.catFile = '%s/iter%d/wfs_catalog.txt' % (self.pertDir, self.iIter)
        wfs.zCompFile = '%s/iter%d/checkZ4C_iter%d.png' % (self.pertDir, self.iIter, self.iIter)

        if iIter>0:
            self.zTrueFile_m1 = '%s/iter%d/sim%d_iter%d_opd.zer' % (
                self.imageDir, self.iIter - 1, self.iSim, self.iIter - 1)
            wfs.zFile_m1 = '%s/iter%d/sim%d_iter%d.z4c' % (
                self.imageDir, self.iIter - 1, self.iSim, self.iIter - 1)
            self.pertMatFile_m1 = '%s/iter%d/sim%d_iter%d_pert.mat' % (
                self.pertDir, self.iIter - 1, self.iSim, self.iIter - 1)
            self.stateV = np.loadtxt(self.pertMatFile_m1)
            
            # PSSN from last iteration needs to be known for shiftGear
            if not (hasattr(metr, 'GQFWHMeff')):
                metr.PSSNFile_m1 = '%s/iter%d/sim%d_iter%d_PSSN.txt'%(
                    self.imageDir, self.iIter - 1, self.iSim, self.iIter - 1)
                aa = np.loadtxt(metr.PSSNFile_m1)
                metr.GQFWHMeff = aa[1, -1] 
                    
    def getOPDAll(self, opdoff, wfs, metr, numproc, wavelength, debugLevel):

        if not opdoff:
            self.writeOPDinst(metr, wavelength)
            self.writeOPDcmd(metr)
            self.OPD_log = '%s/iter%d/sim%d_iter%d_opd%d.log' % (
                self.imageDir, self.iIter, self.iSim, self.iIter, metr.nFieldp4)
    
            if debugLevel >= 3:
                runProgram('head %s' % self.OPD_inst)
                runProgram('head %s' % self.OPD_cmd)
    
            myargs = '%s -c %s -i %s -p %d -e %d > %s' % (
                self.OPD_inst, self.OPD_cmd, self.inst, numproc, self.eimage, self.OPD_log)
            if debugLevel >= 2:
                print('*******Runnnig PHOSIM with following parameters*******')
                print('Check the log file below for progress')
                print('%s' % myargs)
            runProgram('python %s/phosim.py' % self.phosimDir, argstring=myargs)
            if debugLevel >= 3:
                print('DONE RUNNING PHOSIM FOR OPD')
            if os.path.isfile(self.zTrueFile):
                os.remove(self.zTrueFile)
            fz = open(self.zTrueFile, 'ab')
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

    def getOPDAllfromBase(self, baserun, metr):
        self.OPD_inst = '%s/iter%d/sim%d_iter%d_opd%d.inst' % (
            self.pertDir, self.iIter, self.iSim, self.iIter, metr.nFieldp4)
        if not os.path.isfile(self.OPD_inst):
            baseFile = self.OPD_inst.replace('sim%d'%self.iSim, 'sim%d'%baserun)
            os.link(baseFile, self.OPD_inst)

        self.OPD_cmd = '%s/iter%d/sim%d_iter%d_opd%d.cmd' % (
            self.pertDir, self.iIter, self.iSim, self.iIter, metr.nFieldp4)
        if not os.path.isfile(self.OPD_cmd):
            baseFile = self.OPD_cmd.replace('sim%d'%self.iSim, 'sim%d'%baserun)
            os.link(baseFile, self.OPD_cmd)
        
        self.OPD_log = '%s/iter%d/sim%d_iter%d_opd%d.log' % (
            self.imageDir, self.iIter, self.iSim, self.iIter, self.nFieldp4)
        if not os.path.isfile(self.OPD_log):
            baseFile = self.OPD_log.replace('sim%d'%self.iSim, 'sim%d'%baserun)
            os.link(baseFile, self.OPD_log)

        if not os.path.isfile(self.zTrueFile):
            baseFile = self.zTrueFile.replace('sim%d'%self.iSim, 'sim%d'%baserun)
            os.link(baseFile, self.zTrueFile)
        
        for i in range(metr.nFieldp4):
            opdFile = '%s/iter%d/sim%d_iter%d_opd%d.fits' % (
                self.imageDir, self.iIter, self.iSim, self.iIter, i)
            if not os.path.isfile(opdFile):
                baseFile = opdFile.replace('sim%d'%self.iSim, 'sim%d'%baserun)
                os.link(baseFile, opdFile)
                                                                    
    def writeOPDinst(self, metr, wavelength):
        self.OPD_inst = '%s/iter%d/sim%d_iter%d_opd%d.inst' % (
            self.pertDir, self.iIter, self.iSim, self.iIter, metr.nFieldp4)
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

    def writeOPDcmd(self, metr):
        self.OPD_cmd = '%s/iter%d/sim%d_iter%d_opd%d.cmd' % (
            self.pertDir, self.iIter, self.iSim, self.iIter, metr.nFieldp4)
        fid = open(self.OPD_cmd, 'w')
        fid.write('zenith_v 1000.0\n\
raydensity 0.0\n\
perturbationmode 1\n')
        fid.close()

    def getPSFAll(self, psfoff, metr, numproc, debugLevel):

        if not psfoff:
            self.writePSFinst(metr)
            self.writePSFcmd(metr)
            self.PSF_log = '%s/iter%d/sim%d_iter%d_psf%d.log' % (
                self.imageDir, self.iIter, self.iSim, self.iIter, metr.nField)
    
            myargs = '%s -c %s -i %s -p %d -e %d > %s' % (
                self.PSF_inst, self.PSF_cmd, self.inst, numproc, self.eimage, self.PSF_log)
            if debugLevel >= 2:
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
                plt.imshow(extractArray(psf, 20), origin='lower', interpolation='none')
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

    def getPSFAllfromBase(self, baserun, metr):
        self.PSF_inst = '%s/iter%d/sim%d_iter%d_psf%d.inst' % (
            self.pertDir, self.iIter, self.iSim, self.iIter, metr.nField)
        if not os.path.isfile(self.PSF_inst):
            baseFile = self.PSF_inst.replace('sim%d'%self.iSim, 'sim%d'%baserun)
            #PSF files are not crucial, it is ok if the baserun doesn't have it
            if os.path.isfile(baseFile):
                os.link(baseFile, self.PSF_inst)
            else:
                return

        self.PSF_cmd = '%s/iter%d/sim%d_iter%d_psf%d.cmd' % (
            self.pertDir, self.iIter, self.iSim, self.iIter, metr.nField)
        if not os.path.isfile(self.PSF_cmd):
            baseFile = self.PSF_cmd.replace('sim%d'%self.iSim, 'sim%d'%baserun)
            os.link(baseFile, self.PSF_cmd)

        self.PSF_log = '%s/iter%d/sim%d_iter%d_psf%d.log' % (
            self.imageDir, self.iIter, self.iSim, self.iIter, metr.nField)
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
        self.PSF_inst = '%s/iter%d/sim%d_iter%d_psf%d.inst' % (
            self.pertDir, self.iIter, self.iSim, self.iIter, metr.nField)
        fid = open(self.PSF_inst, 'w')
        fid.write('Opsim_filter 1\n\
Opsim_obshistid %d\n\
SIM_VISTIME 15.0\n\
SIM_NSNAP 1\n\
SIM_SEED %d\n\
SIM_CAMCONFIG 1\n'%(self.obsID,  self.obsID%1000-31))
        fpert = open(self.pertFile, 'r')
        fid.write(fpert.read())
        for i in range(metr.nField):
            fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/sed_500.txt 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                i, metr.fieldXp[i], metr.fieldYp[i], self.psfMag))
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
atmosphericdispersion 0\n')
        fid.close()

    def getWFSAll(self, wfs, metr, numproc, debugLevel):

        self.writeWFScmd(wfs, -1)
        for iRun in range(wfs.nRun):
            self.WFS_log = '%s/iter%d/sim%d_iter%d_wfs%d.log' % (
                self.imageDir, self.iIter, self.iSim, self.iIter, wfs.nWFS)
            if wfs.nRun == 1:
                self.writeWFSinst(wfs, metr, -1)
            else:
                self.writeWFSinst(wfs, metr, iRun)
                self.WFS_log = self.WFS_log.replace('.log', '_%s.log'%(wfs.halfChip[iRun]))
                
            myargs = '%s -c %s -i %s -p %d -e %d > %s' % (
                self.WFS_inst, self.WFS_cmd, self.inst, numproc, self.eimage, self.WFS_log)
            if debugLevel >= 2:
                print('********Runnnig PHOSIM with following parameters********')
                print('Check the log file below for progress')
                print('%s' % myargs)

            runProgram('python %s/phosim.py' % self.phosimDir, argstring=myargs)
            plt.figure(figsize=(10, 5))
            for i in range(metr.nFieldp4-wfs.nWFS, metr.nFieldp4):
                chipStr, px, py = self.fieldXY2Chip(
                    metr.fieldXp[i], metr.fieldYp[i], debugLevel)
                src = glob.glob('%s/output/*%s*%s*' % (self.phosimDir, self.obsID, chipStr))
                if wfs.nRun == 1:
                    for ioffset in [0, 1]:
                        if '.gz' in src[0]:
                            runProgram('gunzip -f %s' % src[ioffset])
                        chipFile = src[ioffset].replace('.gz', '')
                        runProgram('mv -f %s %s/iter%d' %( chipFile, self.imageDir, self.iIter))
                else:
                    if '.gz' in src[0]:
                        runProgram('gunzip -f %s' % src[0])
                    chipFile = src[0].replace('.gz', '')
                    targetFile = os.path.split(chipFile.replace('E000', '%s_E000'%wfs.halfChip[iRun]))[1]
                    runProgram('mv -f %s %s/iter%d/%s' %( chipFile, self.imageDir, self.iIter, targetFile))
                    
    def writeWFSinst(self, wfs, metr, iRun=-1):
        #iRun = -1 means only need to run it once
        self.WFS_inst = '%s/iter%d/sim%d_iter%d_wfs%d.inst' % (
            self.pertDir, self.iIter, self.iSim, self.iIter, wfs.nWFS)
        if iRun != -1:
            self.WFS_inst = self.WFS_inst.replace('.inst','_%s.inst'%wfs.halfChip[iRun])
                
        fid = open(self.WFS_inst, 'w')
        fpert = open(self.pertFile, 'r')
        hasCamPiston = False
        for line in fpert:
            if iRun != -1 and line.split()[:2] ==['move', '10']:
                # move command follow Zemax coordinate system.
                fid.write('move 10 %9.4f\n'%(float(line.split()[2])-wfs.offset[iRun]*1e3))
                hasCamPiston = True
            else:
                fid.write(line)
        if  iRun != -1 and (not hasCamPiston):
            fid.write('move 10 %9.4f\n'%(-wfs.offset[iRun]*1e3))
            
        fpert.close()
        fid.write('Opsim_filter 1\n\
Opsim_obshistid %d\n\
SIM_VISTIME 15.0\n\
SIM_NSNAP 1\n\
SIM_SEED %d\n\
Opsim_rawseeing 0.7283\n' % (self.obsID, self.obsID%1000-4))
        if self.inst[:4] == 'lsst':
            fid.write('SIM_CAMCONFIG 2\n')
        elif self.inst[:6] == 'comcam':
            fid.write('SIM_CAMCONFIG 1\n')
            
        ii = 0
        for i in range(metr.nFieldp4-wfs.nWFS, metr.nFieldp4):
            if self.inst[:4] == 'lsst':
                if i % 2 == 1:  # field 31, 33, R44 and R00
                    fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/sed_500.txt 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                        ii, metr.fieldXp[i] + 0.020, metr.fieldYp[i],
                        self.cwfsMag))
                    ii += 1
                    fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/sed_500.txt 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                        ii, metr.fieldXp[i] - 0.020, metr.fieldYp[i],
                        self.cwfsMag))
                    ii += 1
                else:
                    fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/sed_500.txt 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                        ii, metr.fieldXp[i], metr.fieldYp[i] + 0.020,
                        self.cwfsMag))
                    ii += 1
                    fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/sed_500.txt 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                        ii, metr.fieldXp[i], metr.fieldYp[i] - 0.020,
                        self.cwfsMag))
                    ii += 1
            elif self.inst[:6] == 'comcam':
                fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/sed_500.txt 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n' % (
                    ii, metr.fieldXp[i], metr.fieldYp[i],
                    self.cwfsMag))
                ii += 1                    
        fid.close()
        fpert.close()

    def writeWFScmd(self, wfs, iRun=-1):
        #iRun = -1 means only need to run it once
        self.WFS_cmd = '%s/iter%d/sim%d_iter%d_wfs%d.cmd' % (
            self.pertDir, self.iIter, self.iSim, self.iIter, wfs.nWFS)
        if iRun != -1:
            self.WFS_cmd = self.WFS_cmd.replace('.cmd','_%s.cmd'%wfs.halfChip[iRun])
            
        fid = open(self.WFS_cmd, 'w')
        fid.write('zenith_v 1000.0\n\
raydensity 0.0\n\
perturbationmode 1\n\
trackingmode 0\n\
cleartracking\n\
clearclouds\n')
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
