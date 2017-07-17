#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import sys
import numpy as np
import aosCoTransform as ct
from scipy.interpolate import Rbf

from lsst.cwfs.tools import ZernikeAnnularFit
from lsst.cwfs.tools import ZernikeAnnularEval

class aosM1M3(object):

    def __init__(self, debugLevel):
        self.R = 4.180
        self.Ri = 2.558
        self.R3 = 2.508
        self.R3i = 0.550

        self.r1 = -1.9835e4
        self.r3 = -8344.5
        self.k1 = -1.215
        self.k3 = 0.155
        self.alpha1 = np.zeros((8, 1))
        self.alpha1[2] = 1.38e-24
        self.alpha3 = np.zeros((8, 1))
        self.alpha3[2] = -4.5e-22
        self.alpha3[3] = -8.2e-30

        # bending modes
        aa = np.loadtxt('data/M1M3/M1M3_1um_156_grid.DAT')
        self.nodeID = aa[:, 0]
        self.bx = aa[:, 1]
        self.by = aa[:, 2]
        self.bz = aa[:, 3:]
        aa = np.loadtxt('data/M1M3/M1M3_1um_156_force.DAT')
        self.force = aa[:, :]

        if debugLevel >= 3:
            print('-b13--  %f' % self.bx[33])
            print('-b13--  %f' % self.by[193])
            print('-b13--  %d' % np.sum(self.nodeID == 1))
            print('-b13--  %e' % self.bz[332, 15])
            print('-b13--  %e' % self.bz[4332, 15])

        self.bx, self.by, self.bz = ct.M1CRS2ZCRS(self.bx, self.by, self.bz)

        # data needed to determine gravitational print through
        aa = np.loadtxt('data/M1M3/M1M3_dxdydz_zenith.txt')
        self.zdx = aa[:, 0]
        self.zdy = aa[:, 1]
        self.zdz = aa[:, 2]
        aa = np.loadtxt('data/M1M3/M1M3_dxdydz_horizon.txt')
        self.hdx = aa[:, 0]
        self.hdy = aa[:, 1]
        self.hdz = aa[:, 2]
        self.zf = np.loadtxt('data/M1M3/M1M3_force_zenith.txt')
        self.hf = np.loadtxt('data/M1M3/M1M3_force_horizon.txt')
        self.G = np.loadtxt('data/M1M3/M1M3_influence_256.txt')
        self.LUTfile = 'data/M1M3/M1M3_LUT.txt'
        self.nzActuator = 156
        self.nActuator = 256

        # data needed to determine thermal deformation
        aa = np.loadtxt('data/M1M3/M1M3_thermal_FEA.txt', skiprows=1)
        x, y, _ = ct.ZCRS2M1CRS(self.bx, self.by, self.bz)
        # these are normalized coordinates
        # n.b. these may not have been normalized correctly, b/c max(tx)=1.0
        # I tried to go back to the xls data, max(x)=164.6060 in,
        # while 4.18m=164.5669 in.
        tx = aa[:, 0]
        ty = aa[:, 1]
        # below are in M1M3 coordinate system, and in micron
        ip = Rbf(tx, ty, aa[:, 2])
        self.tbdz = ip(x / self.R, y / self.R)
        ip = Rbf(tx, ty, aa[:, 3])
        self.txdz = ip(x / self.R, y / self.R)
        ip = Rbf(tx, ty, aa[:, 4])
        self.tydz = ip(x / self.R, y / self.R)
        ip = Rbf(tx, ty, aa[:, 5])
        self.tzdz = ip(x / self.R, y / self.R)
        ip = Rbf(tx, ty, aa[:, 6])
        self.trdz = ip(x / self.R, y / self.R)

    def idealShape(self, x, y, annulus, dr1=0, dr3=0, dk1=0, dk3=0):
        """
        x,y,and z0 are all in millimeter.
        annulus=1. these (x,y) are on M1 surface
        annulus=3. these (x,y) are on M3 surface
        """
        nr = x.shape
        mr = y.shape
        if (nr != mr):
            print(
                'idealM1M3.m: x is [%d] while y is [%d]. exit. \n' % (nr, mr))
            sys.exit()

        c1 = 1 / (self.r1 + dr1)
        k1 = self.k1 + dk1
        c3 = 1 / (self.r3 + dr3)
        k3 = self.k3 + dk3

        r2 = x**2 + y**2

        idxM1 = annulus == 1
        idxM3 = annulus == 3

        cMat = np.zeros(nr)
        kMat = np.zeros(nr)
        alphaMat = np.tile(np.zeros(nr), (8, 1))
        cMat[idxM1] = c1
        cMat[idxM3] = c3
        kMat[idxM1] = k1
        kMat[idxM3] = k3
        for i in range(8):
            alphaMat[i, idxM1] = self.alpha1[i]
            alphaMat[i, idxM3] = self.alpha3[i]

        # M3 vertex offset from M1 vertex, values from Zemax model
        M3voffset = (233.8 - 233.8 - 900 - 3910.701 - 1345.500 +
                     1725.701 + 3530.500 + 900 + 233.800)

        # ideal surface
        z0 = cMat * r2 / (1 + np.sqrt(1 - (1 + kMat) * cMat**2 * r2))
        for i in range(8):
            z0 = z0 + alphaMat[i, :] * r2**(i + 1)

        z0[idxM3] = z0[idxM3] + M3voffset
        # in Zemax, z axis points from M1M3 to M2. We want z0>0
        return -z0

    def getPrintthz(self, zAngle):
        # M1M3 gravitational and thermal
        printthx = self.zdx * \
            np.cos(zAngle) + self.hdx * np.sin(zAngle)
        printthy = self.zdy * \
            np.cos(zAngle) + self.hdy * np.sin(zAngle)
        printthz = self.zdz * \
            np.cos(zAngle) + self.hdz * np.sin(zAngle)

        # convert dz to grid sag
        # bx, by, bz, written out by senM35pointZMX.m, has been converted
        # to ZCRS, b/c we needed to import those directly into Zemax
        x, y, _ = ct.ZCRS2M1CRS(self.bx, self.by, self.bz)
        # self.idealShape() uses mm everywhere
        zpRef = self.idealShape((x + printthx) * 1000,
                                (y + printthy) * 1000, self.nodeID) / 1000
        zRef = self.idealShape(x * 1000, y * 1000, self.nodeID) / 1000
        # convert printthz into surface sag
        printthz = printthz - (zpRef - zRef)
        zc = ZernikeAnnularFit(printthz, x / self.R,
                               y / self.R, 3, self.Ri / self.R)
        printthz = printthz - ZernikeAnnularEval(
            zc, x / self.R, y / self.R, self.Ri / self.R)
        return printthz
    
    
