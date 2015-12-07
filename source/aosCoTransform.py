#!/usr/bin/env python
##
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope


def M1CRS2ZCRS(x, y, z):
    """ from M1 CRS to Zemax CRS"""
    return -x, y, -z


def M2CRS2ZCRS(x, y, z):
    """ from M2 CRS to Zemax CRS"""
    return -x, y, -z
