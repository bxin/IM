#!/usr/bin/env python

# @author: Bo Xin
# @      Large Synoptic Survey Telescope

import os, shutil, glob
import subprocess

import numpy as np
from astropy.io import fits

from cwfsTools import ZernikeAnnularFit
from cwfsTools import extractArray

import matplotlib.pyplot as plt

class aosTeleState(object):
    def __init__(self,esti, instruFile, iSim, phosimDir,
                 pertDir, imageDir, debugLevel):
        #plan to write these to txt files. no columns for iter
        uk=np.zeros(esti.ndofA) #to be added
        pertuk=np.zeros(esti.ndofA) # telescope state(?)
        
        self.filename = os.path.join('data/', (instruFile + '.inst'))
        fid = open(self.filename)
        iscomment = False
        for line in fid:
            line = line.strip()
            if (line.startswith('###')):
                iscomment = ~iscomment
            if (not(line.startswith('#')) and
                    (not iscomment) and len(line) > 0):
                if (line.startswith('idof')):
                    self.idof = int(line.split()[1])
                    arrayType = 'idof'
                    arrayCount = 0
                elif (line.startswith('opd_size')):
                    self.opdSize = int(line.split()[1])
                    if self.opdSize%2 ==0:
                        self.opdSize -= 1
                elif (line.startswith('eimage')):
                    self.eimage = bool(int(line.split()[1]))
                elif (line.startswith('psf_mag')):
                    self.psfMag = int(line.split()[1])
                elif (line.startswith('psf_stamp_size')):
                    self.psfStampSize = int(line.split()[1])
                else:
                    line1=line.replace('1','1 ')
                    line1=line1.replace('0','0 ')
                    if (line1[0].isdigit()):
                        arrayCount = arrayCount + 1
                    if (arrayType == 'idof' and arrayCount == self.idof):
                        self.dofIdx = np.fromstring(line1,dtype=bool,sep=' ')
                        arrayCount = 0
                        arrayType = ''

        fid.close()

        self.iSim = iSim
        self.iter = 0
        self.phosimDir = phosimDir
        self.pertDir = pertDir
        self.imageDir = imageDir
        self.pertFile='%s/sim%d_iter0_pert.txt'%(self.pertDir, iSim)
        if self.idof>0:
            fid = open(self.pertFile, 'w')
            for i in range(len(self.dofIdx)):
                if self.dofIdx[i]:
                    fid.write('bending %d 1um\n'%i)
            fid.close()
            
        if debugLevel>=3:
            print('in aosTeleState:')
            print(self.dofIdx)

    def setIterNo(self, iiter):
        self.iiter = iiter
        
    def getOPD35(self, wfs, metr, numproc, wavelength, debugLevel):

        self.OPD_inst = '%s/sim%d_iter%d_opd35.inst'%(
            self.pertDir, self.iSim,self.iiter)
        fid = open(self.OPD_inst, 'w')
        fid.write('SIM_VISTIME 15.0\n\
SIM_NSNAP 1\n')
        fpert = open(self.pertFile, 'r')
        fid.write(fpert.read())
        for i in range(metr.nFieldp4):
            fid.write('opd %2d\t%9.6f\t%9.6f %5.1f\n'%(
                i,metr.fieldX[i],metr.fieldY[i],wavelength*1e3))
        fid.close()
        fpert.close()

        self.opdGrid1d=np.linspace(-1,1,self.opdSize)
        self.opdx, self.opdy = np.meshgrid(self.opdGrid1d, self.opdGrid1d)

        myargs='%s -c %s/examples/lsst_aos/opd_command -p %d -e %d'%(
            self.OPD_inst, self.phosimDir, numproc,self.eimage)
        runProgram('python %s/phosim.py'%self.phosimDir, argstring=myargs)
        zFile =  '%s/sim%d_iter%d_opd.zer'%(
            self.imageDir, self.iSim,self.iiter)
        if os.path.isfile(zFile):
            os.remove(zFile)
        fz = open(zFile, 'ab')
        for i in range(metr.nFieldp4):
            src = '%s/output/opd_%d.fits.gz'%(self.phosimDir,i)
            dst = '%s/sim%d_iter%d_opd%d.fits.gz'%(
                self.imageDir, self.iSim,self.iiter,i)
            shutil.move(src, dst)
            runProgram('gunzip -f %s'%dst)
            opdFile = dst.replace('.gz','')
            IHDU = fits.open(opdFile)
            opd = IHDU[0].data*1e3 #from mm to um
            IHDU.close()
            idx = (opd !=0)
            Z = ZernikeAnnularFit(opd[idx], self.opdx[idx], self.opdy[idx],
                                  wfs.znwcs, wfs.inst.obscuration)
            np.savetxt(fz, Z.reshape(1,-1), delimiter= ' ')

        fz.close()
        
        if debugLevel>=3:
            print(self.opdGrid1d.shape)
            print(self.opdGrid1d[0])
            print(self.opdGrid1d[-1])
            print(self.opdGrid1d[-2])
            print(self.opdx)
            print(self.opdy)
            print(wfs.znwcs)
            print(wfs.inst.obscuration)
            
    def getPSF31(self, metr, numproc, debugLevel):
        
        self.PSF_inst = '%s/sim%d_iter%d_psf31.inst'%(
            self.pertDir, self.iSim,self.iiter)
        fid = open(self.PSF_inst, 'w')
        fid.write('SIM_VISTIME 15.0\n\
SIM_NSNAP 1\n')
        fpert = open(self.pertFile, 'r')
        fid.write(fpert.read())
        for i in range(metr.nField):
            fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/sed_flat.txt 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n'%(
                i,metr.fieldXp[i],metr.fieldYp[i],self.psfMag))
        fid.close()
        fpert.close()

        myargs='%s -c %s/examples/lsst_aos/psf_command -p %d -e %d'%(
            self.PSF_inst, self.phosimDir, numproc, self.eimage)
#        runProgram('python %s/phosim.py'%self.phosimDir, argstring=myargs)
        fig = plt.figure(figsize=(10, 10))
        for i in range(metr.nField):
            chipStr, px, py = fieldXY2Chip(metr.fieldXp[i],metr.fieldYp[i],
                                   self.phosimDir, debugLevel)
            src = glob.glob('%s/output/*%s*'%(self.phosimDir,chipStr))
            if 'gz' in src[0]:
                runProgram('gunzip -f %s'%src[0])
            IHDU = fits.open(src[0].replace('.gz',''))
            chipImage = IHDU[0].data
            IHDU.close()
            psf = chipImage[py-self.psfStampSize/2:py+self.psfStampSize/2,
                            px-self.psfStampSize/2:px+self.psfStampSize/2]
            offsety = np.argwhere(psf==psf.max())[0][0] - \
              self.psfStampSize/2+1
            offsetx = np.argwhere(psf==psf.max())[0][1] - \
              self.psfStampSize/2+1
            psf = chipImage[
                py-self.psfStampSize/2+offsety:py+self.psfStampSize/2+offsety,
                px-self.psfStampSize/2+offsetx:px+self.psfStampSize/2+offsetx]

            if i==0:
                pIdx=1
            else:
                pIdx=i+metr.nArm

            dst = '%s/sim%d_iter%d_psf%d.fits'%(
                self.imageDir, self.iSim,self.iiter,i)
            if os.path.isfile(dst):
                os.remove(dst)
            hdu = fits.PrimaryHDU(psf)
            hdu.writeto(dst)
            
            ax = plt.subplot(metr.nRing+1, metr.nArm, pIdx)
            plt.imshow(extractArray(psf,20), origin='lower')
            plt.title('%d'%i)
            plt.axis('off')
            
            if debugLevel>=3:
                print('px = %d, py = %d'%(px,py))
                print('offsetx = %d, offsety = %d'%(offsetx,offsety))
                print('passed %d'%i)
                
        #plt.show()
        pngFile = '%s/sim%d_iter%d_psf.png'%(
                self.imageDir, self.iSim,self.iiter)        
        plt.savefig(pngFile,bbox_inches='tight')
                    
    def getWFS4(self, metr, numproc, debugLevel):
        self.WFS_inst = '%s/sim%d_iter%d_wfs4.inst'%(
            self.pertDir, self.iSim,self.iiter)
        fid = open(self.WFS_inst, 'w')
        fid.write('SIM_VISTIME 15.0\n\
SIM_NSNAP 1\n')
        fpert = open(self.pertFile, 'r')
        fid.write(fpert.read())
        ii = 0
        for i in range(metr.nField, metr.nFieldp4):
            if i%2==0:
                fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/sed_flat.txt 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n'%(
                ii,metr.fieldXp[i]+0.004,metr.fieldYp[i],self.psfMag))
                ii += 1
                fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/sed_flat.txt 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n'%(
                ii,metr.fieldXp[i]-0.004,metr.fieldYp[i],self.psfMag))
                ii += 1
            else:
                fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/sed_flat.txt 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n'%(
                ii,metr.fieldXp[i],metr.fieldYp[i]+0.004,self.psfMag))
                ii += 1
                fid.write('object %2d\t%9.6f\t%9.6f %9.6f \
../sky/sed_flat.txt 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0  none  none\n'%(
                ii,metr.fieldXp[i],metr.fieldYp[i]-0.004,self.psfMag))
                ii += 1
        fid.close()
        fpert.close()
        exit()
        myargs='%s -c %s/examples/lsst_aos/psf_command -p %d -e %d'%(
            self.WFS_inst, self.phosimDir, numproc, self.eimage)
        runProgram('python %s/phosim.py'%self.phosimDir, argstring=myargs)
        fig = plt.figure(figsize=(5, 10))
        for i in range(metr.nField, metr.nFieldp4):
            chipStr, px, py = fieldXY2Chip(metr.fieldXp[i],metr.fieldYp[i],
                                   self.phosimDir, debugLevel)
            src = glob.glob('%s/output/*%s*'%(self.phosimDir,chipStr))
            if 'gz' in src[0]:
                runProgram('gunzip -f %s'%src[0])
            IHDU = fits.open(src[0].replace('.gz',''))
            chipImage = IHDU[0].data
            IHDU.close()
            psf = chipImage[py-self.psfStampSize/2:py+self.psfStampSize/2,
                            px-self.psfStampSize/2:px+self.psfStampSize/2]
            offsety = np.argwhere(psf==psf.max())[0][0] - \
              self.psfStampSize/2+1
            offsetx = np.argwhere(psf==psf.max())[0][1] - \
              self.psfStampSize/2+1
            psf = chipImage[
                py-self.psfStampSize/2+offsety:py+self.psfStampSize/2+offsety,
                px-self.psfStampSize/2+offsetx:px+self.psfStampSize/2+offsetx]

            if i==0:
                pIdx=1
            else:
                pIdx=i+metr.nArm

            dst = '%s/sim%d_iter%d_psf%d.fits'%(
                self.imageDir, self.iSim,self.iiter,i)
            if os.path.isfile(dst):
                os.remove(dst)
            hdu = fits.PrimaryHDU(psf)
            hdu.writeto(dst)
            
            ax = plt.subplot(metr.nRing+1, metr.nArm, pIdx)
            plt.imshow(extractArray(psf,20), origin='lower')
            plt.title('%d'%i)
            plt.axis('off')
            
            if debugLevel>=3:
                print('px = %d, py = %d'%(px,py))
                print('offsetx = %d, offsety = %d'%(offsetx,offsety))
                print('passed %d'%i)
                
        #plt.show()
        pngFile = '%s/sim%d_iter%d_psf.png'%(
                self.imageDir, self.iSim,self.iiter)        
        plt.savefig(pngFile,bbox_inches='tight')
                    
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
            mydict[line.split()[0]] = [float(line.split()[1]), float(line.split()[2])]
            
    f.close()
    ruler = sorted(set([x[0] for x in mydict.values()]))
    return ruler

def fieldXY2Chip(fieldX, fieldY, phosimDir, debugLevel):
    ruler = getChipBoundary(
        '%s/data/lsst/focalplanelayout.txt'%phosimDir)

    rx,cx,px = fieldAgainstRuler(ruler, fieldX, 4000)
    ry,cy,py = fieldAgainstRuler(ruler, fieldY, 4072)
    
    if debugLevel>=3:
        print(ruler)
        print(len(ruler))

    return 'R%d%d_S%d%d'%(rx,ry,cx,cy), px, py
        
def fieldAgainstRuler(ruler, field, chipPixel):

    field = field*180000 #degree to micron
    p2 = (ruler>=field)
    if (np.count_nonzero(p2) == 0):  #  too large to be in range
        p = len(ruler)-1 #p starts from 0
    elif (p2[0]):
        p = 0
    else:
        p1 = p2.argmax() - 1
        p2 = p2.argmax()
        if (ruler[p2] - field)<(field - ruler[p1]):
            p = p2
        else:
            p = p1
            
    pixel = (field - ruler[p])/10 # 10 for 10micron pixels
    pixel += chipPixel/2
            
    return np.floor(p/3), p%3, pixel


        
