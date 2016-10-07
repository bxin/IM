#!/usr/bin/env python
##
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope


def M1CRS2ZCRS(x, y, z):
    """ from M1 CRS to Zemax CRS"""
    return -x, y, -z

def ZCRS2M1CRS(x, y, z):
    """ from Zemax CRS to M1 CRS"""
    return -x, y, -z

def M2CRS2ZCRS(x, y, z):
    """ from M2 CRS to Zemax CRS"""
    return -x, y, -z

def ZCRS2M2CRS(x, y, z):
    """ from Zemax CRS to M2 CRS"""
    return -x, y, -z
