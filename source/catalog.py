import numpy as np
from astropy.table import join, Table, vstack


class Catalog(object):

    def __init__(self, table=None):
        self.__sourceIdCounter = 0
        if table:
            self.table = table
        else:
            self.table = Table(names=('sourceId', 'ra', 'dec', 'mag', 'sed'),
                           dtype=('i8', 'f16', 'f16', 'f8', 'S'))

    def addSource(self, ra, dec, mag, sed):
        self.table.add_row((self.__sourceIdCounter, ra, dec, mag, sed))
        self.__sourceIdCounter += 1

    def toFile(self, fname):
        self.table.write(fname, format='csv')

    @classmethod
    def fromFile(cls, fname):
        table = Table.read(fname, format='csv')
        return cls(table=table)