import numpy as np
from astropy.table import Table

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

    def getPhosimBody(self):
        template = 'object {} {} {} {} ../sky/{} ' \
               '0.0 0.0 0.0 0.0 0.0 0.0 star 0.0 none none\n'
        lines = [template.format(sourceId, ra, dec, mag, sed) for sourceId, ra, dec, mag, sed in
             self.table]
        return ''.join(lines)

    @classmethod
    def fromFile(cls, fname):
        table = Table.read(fname, format='csv')
        return cls(table=table)


class GridCatalog(Catalog):
    """
    Used to make a grid of nxn sources on each listed chip.
    Assumes boresight is (0,0) and rotation is 0.
    """
    def __init__(self, n=5, chips=['R00_S22', 'R04_S20', 'R40_S02', 'R44_S00'], mag=17,
                 sed='../sky/sed_500.txt'):
        super().__init__(self)

        # lazy imports
        from lsst.obs.lsst.phosim import PhosimMapper
        from lsst.afw.cameraGeom import PIXELS, FIELD_ANGLE
        from lsst.afw.geom import Point2D

        camera = PhosimMapper().camera

        for chip in chips:
            det = camera[chip]
            cornersPix = [Point2D(x) for x in det.getBBox().getCorners()]
            pixels2Field = camera.getTransform(det.makeCameraSys(PIXELS), FIELD_ANGLE)
            corners = pixels2Field.applyForward(cornersPix)
            ras = np.rad2deg([c.getX() for c in corners])
            decs = np.rad2deg([c.getY() for c in corners])

            for ra in np.linspace(min(ras), max(ras), n + 2)[1:-1]:
                for dec in np.linspace(min(decs), max(decs), n + 2)[1:-1]:
                    self.addSource(ra, dec, mag, sed)
