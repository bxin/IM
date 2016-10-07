#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import numpy as np
import aosCoTransform as ct


class aosM2(object):

    def __init__(self, debugLevel):
        self.R = 1.760

        # bending modes
        aa = np.loadtxt('data/M2/M2_1um_grid.DAT')
        self.bx = aa[:, 0]
        self.by = aa[:, 1]
        self.bz = aa[:, 2:]
        # !!! we are using M1M3 forces in place of M2 forces
        aa = np.loadtxt('data/M2/M2_1um_force.DAT')
        self.force = aa[:, :]

        if debugLevel >= 3:
            print('-b2--  %f' % self.bx[33])
            print('-b2--  %f' % self.by[193])
            print('-b2--  %d' % self.bx.shape)
            print('-b2--  %e' % self.bz[332, 15])
            print('-b2--  %e' % self.bz[4332, 15])

        self.bx, self.by, self.bz = ct.M2CRS2ZCRS(self.bx, self.by, self.bz)

        self.bxnorm = self.bx / self.R
        self.bynorm = self.by / self.R
