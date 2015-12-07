#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import numpy as np
import aosCoTransform as ct


class aosM1M3(object):

    def __init__(self, debugLevel):
        self.R = 4.180

        # bending modes
        aa = np.loadtxt('data/bendingModes/M1M3_1um_grid.DAT')
        self.ID = aa[:, 0]
        self.bx = aa[:, 1]
        self.by = aa[:, 2]
        self.bz = aa[:, 3:]
        aa = np.loadtxt('data/bendingModes/M1M3_1um_force.DAT')
        self.force = aa[:, :]

        if debugLevel >= 3:
            print('-b13--  %f' % self.bx[33])
            print('-b13--  %f' % self.by[193])
            print('-b13--  %d' % np.sum(self.ID == 1))
            print('-b13--  %e' % self.bz[332, 15])
            print('-b13--  %e' % self.bz[4332, 15])

        self.bx, self.by, self.bz = ct.M1CRS2ZCRS(self.bx, self.by, self.bz)

        self.bxnorm = self.bx / self.R
        self.bynorm = self.by / self.R
