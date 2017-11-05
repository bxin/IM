#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import os
import numpy as np
import aosCoTransform as ct


class aosM2(object):

    def __init__(self, debugLevel):
        self.R = 1.710 #clear aperture
        self.Ri = 0.9
        # bending modes
        aosSrcDir = os.path.split(os.path.abspath(__file__))[0]
        aa = np.loadtxt('%s/../data/M2/M2_1um_grid.DAT'%aosSrcDir)
        self.bx = aa[:, 0]
        self.by = aa[:, 1]
        self.bz = aa[:, 2:]
        # !!! we are using M1M3 forces in place of M2 forces
        aa = np.loadtxt('%s/../data/M2/M2_1um_force.DAT'%aosSrcDir)
        self.force = aa[:, :]

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
        # should be the same as the x,y converted from the bending modes x and
        # y above
        self.zdz = aa[:, 2]
        self.hdz = aa[:, 3]
        self.tzdz = aa[:, 4]
        self.trdz = aa[:, 5]

    def getPrintthz(self, zAngle):
        printthz = self.zdz * np.cos(zAngle) \
          + self.hdz * np.sin(zAngle)
        pre_comp_elev = 0
        printthz -= self.zdz * np.cos(pre_comp_elev) \
          + self.hdz * np.sin(pre_comp_elev)
        return printthz
    
