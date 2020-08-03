#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import os
import numpy as np
import aosCoTransform as ct
from scipy.interpolate import Rbf

class aosM2(object):

    def __init__(self, debugLevel):
        self.R = 1.710 #clear aperture
        self.Ri = 0.9
        # bending modes
        aosSrcDir = os.path.split(os.path.abspath(__file__))[0]
        aa = np.loadtxt('%s/../data/M2/M2_1um_grid.DAT'%aosSrcDir)
        #first column are FEA node ID numbers
        self.bx = aa[:, 1]
        self.by = aa[:, 2]
        self.bz = aa[:, 3:]

        # M2 forces based on Harris FEA model
        aa = np.loadtxt('%s/../data/M2/M2_1um_force.DAT'%aosSrcDir)
        self.actID = aa[:, 0]
        self.actx = aa[:, 1]
        self.acty = aa[:, 2]
        self.force = aa[:, 3:]

        if debugLevel >= 3:
            print('-b2--  %f' % self.bx[33])
            print('-b2--  %f' % self.by[193])
            print('-b2--  %d' % self.bx.shape)
            print('-b2--  %e' % self.bz[332, 15])
            print('-b2--  %e' % self.bz[4332, 15])

        self.bx, self.by, self.bz = ct.M2CRS2ZCRS(self.bx, self.by, self.bz)

        # M2 gravitational and thermal deformations
        aa = np.loadtxt('%s/../data/M2/M2_GT_FEA.txt'%aosSrcDir, skiprows=1)
        x, y, _ = ct.ZCRS2M2CRS(self.bx, self.by, self.bz)
        # first two columns are normalized x and y,
        # need to interpolate to the bending modes x and
        # y above
        tx = aa[:, 0]
        ty = aa[:, 1]
        # below are in M2 coordinate system, and in micron
        ip = Rbf(tx, ty, aa[:, 2])
        self.zdz = ip(x /self.R, y/self.R)
        ip = Rbf(tx, ty, aa[:, 3])
        self.hdz = ip(x /self.R, y/self.R)
        ip = Rbf(tx, ty, aa[:, 4])
        self.tzdz = ip(x /self.R, y/self.R)
        ip = Rbf(tx, ty, aa[:, 5])
        self.trdz = ip(x /self.R, y/self.R)

    def getPrintthz(self, zAngle):
        printthz = self.zdz * np.cos(zAngle) \
          + self.hdz * np.sin(zAngle)
        pre_comp_elev = 0
        printthz -= self.zdz * np.cos(pre_comp_elev) \
          + self.hdz * np.sin(pre_comp_elev)
        return printthz
    
