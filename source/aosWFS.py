#!/usr/bin/env python
##
# @authors: Bo Xin
# @       Large Synoptic Survey Telescope

import os
import multiprocessing
import re
import aosTeleState

import numpy as np
from astropy.io import fits
from astropy.table import join, Table
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn.apionly as sns

from lsst.cwfs.algorithm import Algorithm
from lsst.cwfs.instrument import Instrument
from lsst.cwfs.image import Image, readFile


class aosWFS(object):

    ZS = ['z{}'.format(i) for i in range(4, 23)]

    def __init__(self, cwfsDir, imageDir, instruFile, algoFile, iSim,
                 imgSizeinPix, band, wavelength, debugLevel):
        self.imageDir = imageDir
        self.obsId = None
        self.iSim = iSim
        self.band = band
        self.iIter = 0
        self.nWFS = 4
        self.nRun = 1
        self.nExp = 1
        self.wfsName = ['intra', 'extra']
        self.offset = [-1.5, 1.5]  # default offset
        self.halfChip = ['C0', 'C1']  # C0 is always intra, C1 is extra

        self.cwfsDir = cwfsDir
        self.inst = Instrument(instruFile, imgSizeinPix)
        self.algo = Algorithm(algoFile, self.inst, debugLevel)
        self.znwcs = self.algo.numTerms
        self.znwcs3 = self.znwcs - 3
        self.myZn = np.zeros((self.znwcs3 * self.nWFS, 2))
        self.trueZn = np.zeros((self.znwcs3 * self.nWFS, 2))
        aa = instruFile
        if aa[-2:].isdigit():
            aa = aa[:-2]
        aosSrcDir = os.path.split(os.path.abspath(__file__))[0]
        intrinsicFile = '%s/../data/%s/intrinsic_zn.txt' % (aosSrcDir, aa)
        if np.abs(wavelength - 0.5)>1e-3:
            intrinsicFile = intrinsicFile.replace(
                'zn.txt', 'zn_%s.txt' % band.upper())
        intrinsicAll = np.loadtxt(intrinsicFile)
        intrinsicAll = intrinsicAll * wavelength
        self.intrinsicWFS = intrinsicAll[
            -self.nWFS:, 3:self.algo.numTerms].reshape((-1, 1))
        self.covM = np.loadtxt('%s/../data/covM86.txt'% aosSrcDir)  # in unit of nm^2
        self.covM = self.covM * 1e-6  # in unit of um^2

        if debugLevel >= 3:
            print('znwcs3=%d' % self.znwcs3)
            print(self.intrinsicWFS.shape)
            print(self.intrinsicWFS[:5])

    def setIterNo(self, iIter):
        self.obsId = 9000000 + self.iSim * 1000 + iIter * 10
        self.iIter = iIter

    def findCandidates(self, catalog):
        centroids = self.getPhosimCentroid()
        candidates = join(catalog.table, centroids, keys=['sourceId'], join_type='inner')
        candidates.sort('sourceId')
        return candidates

    def selectPairs(self, candidates):
        # wavefront sensors
        chips = 'R00_S22 R40_S02 R04_S20 R44_S00'.split()

        pairs = Table(names=['chip', 'intraSourceId', 'extraSourceId'],
                            dtype=['S', 'i8', 'i8'])

        # one sensor at a time
        for chip in chips:
            intraChip = chip + '_C0'
            extraChip = chip + '_C1'
            intraIdx = (candidates['halfchip'] == intraChip)
            extraIdx = (candidates['halfchip'] == extraChip)
            intraDonuts = candidates[intraIdx].copy(copy_data=True)
            extraDonuts = candidates[extraIdx].copy(copy_data=True)

            # ranking algorithm
            intraDonuts.sort('mag')
            extraDonuts.sort('mag')

            # pair brightest donuts until run out
            while len(intraDonuts) > 0 and len(extraDonuts) > 0:
                sourceIdIntra = intraDonuts['sourceId'][0]
                sourceIdExtra = extraDonuts['sourceId'][0]
                pairs.add_row((chip, sourceIdIntra, sourceIdExtra))

                intraDonuts.remove_row(0)
                extraDonuts.remove_row(0)

        return pairs

    def prepareArgList(self, pairs, candidates, cwfsModel):
        argList = []
        for pair in pairs:
            chip = pair['chip']
            intraChip = chip + '_C0'
            extraChip = chip + '_C1'
            intraSourceId = pair['intraSourceId']
            extraSourceId = pair['extraSourceId']

            intraCandidate = candidates[candidates['sourceId'] == intraSourceId][0]
            extraCandidate = candidates[candidates['sourceId'] == extraSourceId][0]

            intraPixX = intraCandidate['pixX']
            intraPixY = intraCandidate['pixY']
            extraPixX = extraCandidate['pixX']
            extraPixY = extraCandidate['pixY']

            intraCrop = self.getCrop(intraChip, intraPixX, intraPixY, widthPix=128)
            extraCrop = self.getCrop(extraChip, extraPixX, extraPixY, widthPix=128)

            # Necessary to reconcile phosim fits files axes with focal plane layout.
            intraCrop = self.rotateByChip(chip, intraCrop)
            extraCrop = self.rotateByChip(chip, extraCrop)

            # This assumes that the boresight is (0,0).
            intraFieldX = intraCandidate['ra']
            intraFieldY = intraCandidate['dec']
            extraFieldX = extraCandidate['ra']
            extraFieldY = extraCandidate['dec']

            intraImage = Image(intraCrop, (extraFieldX, extraFieldY), 'intra')
            extraImage = Image(extraCrop, (intraFieldX, intraFieldY), 'extra')

            argList.append((self.algo, chip, intraSourceId, extraSourceId, intraImage, extraImage,
                            self.inst,
                            cwfsModel))
        return argList

    def processPairs(self, pairs, candidates, cwfsModel, numProc):
        # This argList is necessary for multiprocessing.
        argList = self.prepareArgList(pairs, candidates, cwfsModel)
        pool = multiprocessing.Pool(numProc)
        parallelOutput = pool.map(aosWFS.runcwfs, argList)
        pool.close()
        pool.join()

        # Consolidate parallel output into a table.
        zernikes = Table(names=['chip', 'intraSourceId', 'extraSourceId', 'caustic'] + aosWFS.ZS,
                         dtype=['S', 'i4', 'i4', 'i4'] + ['f4'] * len(aosWFS.ZS))
        for row in parallelOutput:
            chip = row[0]
            intraSourceId = row[1]
            extraSourceId = row[2]
            caustic = row[3]
            z = row[4]
            zernikes.add_row((chip, intraSourceId, extraSourceId, caustic, *z))

        return argList, zernikes

    def makeMasterZernikes(self, candidates, zernikes):
        aggZernikes = zernikes[['chip'] + aosWFS.ZS].group_by('chip').groups.aggregate(np.mean)
        caustics = zernikes['chip', 'caustic'].group_by('chip').groups.aggregate(np.sum)
        masterZernikes = join(aggZernikes, caustics, keys=['chip'], join_type='inner')
        return masterZernikes

    def parallelCwfs(self, catalog, cwfsModel, numproc, debugLevel):
        candidates = self.findCandidates(catalog)
        pairs = self.selectPairs(candidates)
        argList, zernikes = self.processPairs(pairs, candidates, cwfsModel, numproc)
        masterZernikes = self.makeMasterZernikes(candidates, zernikes)

        # Plan to update io so row ordering wont matter.
        oldOut = np.array([
            aosWFS.rowToZernikesAndCaustic(masterZernikes[masterZernikes['chip'] == 'R44_S00'][0]),
            aosWFS.rowToZernikesAndCaustic(masterZernikes[masterZernikes['chip'] == 'R04_S20'][0]),
            aosWFS.rowToZernikesAndCaustic(masterZernikes[masterZernikes['chip'] == 'R00_S22'][0]),
            aosWFS.rowToZernikesAndCaustic(masterZernikes[masterZernikes['chip'] == 'R40_S02'][0]),
        ])

        if debugLevel > 0:
            self.writeTable(candidates, 'candidates.csv')
            self.writeTable(pairs, 'pairs.csv')
            # self.plotPairing(candidates, pairs, 'pairing.png')
            self.plotDonutsAndZernikes(argList, zernikes, 'donutsAndZernikes.png')
            self.writeTable(zernikes, 'zernikes.csv')
            # self.plotMasterZernikes(masterZernikes, 'masterZernikes')

        np.savetxt(self.zFile, oldOut)

    def writeTable(self, table, fname):
        imgDir = self.getCurrentImagePath()
        path = os.path.join(imgDir, fname)
        table.write(path, format='csv')

    @staticmethod
    def rowToZernikes(row):
        arr = [row[key] for key in aosWFS.ZS]
        return np.array(arr)

    # hoping to remove once we upgrade io
    @staticmethod
    def rowToZernikesAndCaustic(row):
        arr = [row[key] for key in aosWFS.ZS + ['caustic']]
        return np.array(arr)

    # def plotPairing(self, candidates, pairs, zernikes, fname):
    #     chips = np.unique(pairs['chip'])
    #     nChips = len(chips)
    #     pltRows = 4
    #     for chip in chips:
    #         intraBoxes = []
    #         extraBoxes = []
    #
    #         chipPairs = pairs[pairs['chip'] == chip]
    #         nPairs = len(chipPairs)
    #         palette = sns.palplot(sns.color_palette("husl", nPairs))
    #         for i,row in enumerate(chipPairs):
    #             intraSourceId = row['intraSourceId']
    #             extraSourceId = row['extraSourceId']
    #
    #             intraCandidate = candidates[candidates['sourceId'] == intraSourceId][0]
    #             extraCandidate = candidates[candidates['sourceId'] == extraSourceId][0]
    #
    #             intraPixX = intraCandidate['pixX']
    #             intraPixY = intraCandidate['pixY']
    #             extraPixX = extraCandidate['pixX']
    #             extraPixY = extraCandidate['pixY']
    #
    #             width = 128
    #             halfwidth = width // 2
    #
    #             # Also switch x,y to be consistent with image.
    #             intraCorner = (intraPixY - halfwidth, intraPixX - halfwidth)
    #             extraCorner = (extraPixY - halfwidth, extraPixX - halfwidth)
    #
    #             intraBoxes.append(Rectangle(intraCorner, width, width, fill=False,
    #                                         color=palette[i]))
    #             extraBoxes.append(Rectangle(extraCorner, width, width, fill=False,
    #                                         color=palette[i]))
    #
    #         plt.subplot(nChips, pltRows, i+1)
    #         plt.title()
    #         halfchip = chip + '_C0'
    #         img = self.getImage(halfchip)
    #         plt.title(halfchip)
    #         cb = plt.imshow(img, cmap='hot', origin='lower')
    #         plt.colorbar(cb)
    #         for box in intraBoxes:
    #             plt.gca().add_patch(box)
    #
    #         plt.subplot(nChips, pltRows, i+2)
    #         halfchip = chip + '_C1'
    #         img = self.getImage(halfchip)
    #
    #         plt.title(halfchip)
    #         plt.imshow()
    #
    #         plt.subplot(nChips, pltRows, i+3)
    #         plt.imshow()
    #
    #         plt.subplot(nChips, pltRows, i+4)
    #         plt.imshow()
    #
    #         self.getImage()
    #         path = os.path.join(self.getCurrentImagePath(), fname)
    #         plt.savefig(path)

    def plotDonutsAndZernikes(self, argList, zernikes, fname):
        nPairs = len(argList)
        plt.figure(figsize=(8.5, nPairs * 2.5))
        zChipTable = zernikes[['chip'] + self.ZS].group_by('chip').groups.aggregate(np.mean)
        zAll = aosWFS.rowToZernikes(zernikes[self.ZS].groups.aggregate(np.mean))

        for i,args in enumerate(argList):
            _, chip, intraSourceId, extraSourceId, intraImage, extraImage, _, _  = args
            plt.subplot(nPairs,3,i*3+1)
            plt.title('{}, {}, Intra'.format(chip, intraSourceId), fontsize=8)
            cb = plt.imshow(intraImage.image, origin='lower', cmap='hot')
            plt.colorbar(cb)
            plt.axis('off')

            plt.subplot(nPairs,3,i*3+2)
            plt.title('{}, {}, Extra'.format(chip, extraSourceId), fontsize=8)
            cb = plt.imshow(extraImage.image, origin='lower', cmap='hot')
            plt.colorbar(cb)
            plt.axis('off')

            plt.subplot(nPairs,3,i*3+3)
            plt.title('Zernikes', fontsize=8)
            zPair = aosWFS.rowToZernikes(zernikes[i])
            zChip = aosWFS.rowToZernikes(zChipTable[zChipTable['chip'] == chip][0])
            zDomain = range(4, 23)
            plt.plot(zDomain, zPair, marker='s', linestyle='--', label='Pair', color='#4286f4',
                     alpha=0.7)
            plt.plot(zDomain, zChip, marker='o', linestyle='--', label=chip, color='#fc41a5',
                     alpha=0.7)
            plt.plot(zDomain, zAll, marker='x', linestyle='--', label='All', color='#3ef979',
                     alpha=0.7)
            plt.grid(b=True)
            plt.legend()

        path = os.path.join(self.getCurrentImagePath(), fname)
        plt.tight_layout()
        plt.savefig(path)

    @staticmethod
    def runcwfs(args):
        algo, chip, intraSourceId, extraSourceId, intraImage, extraImage, inst, model = args
        algo.reset(intraImage, extraImage)
        algo.runIt(inst, intraImage, extraImage, model)
        return chip, intraSourceId, extraSourceId, algo.caustic, algo.zer4UpNm * 1e-3

    def getPhosimCentroid(self):
        centroids = Table(names=['halfchip', 'sourceId', 'nphoton', 'pixX', 'pixY'],
                          dtype=['S', 'i4', 'f4', 'i4', 'i4'])

        # example centroid file: centroid_lsst_e_9018000_f1_R00_S22_C1_E000.txt
        target = 'centroid_lsst_e_\d+_f\d_(R\d{2}_S\d{2}_C\d)_E000.txt'
        pattern = re.compile(target)

        imgPath = self.getCurrentImagePath()
        for fname in os.listdir(imgPath):
            match = pattern.match(fname)
            if match:
                halfchip = match.group(1)
                data = np.loadtxt(os.path.join(imgPath, fname), skiprows=1).reshape(-1, 4)
                for row in data:
                    centroids.add_row((halfchip, *row))
        return centroids

    def getCurrentImagePath(self):
        path = '{}/iter{}'.format(self.imageDir, self.iIter)
        return path

    def getImage(self, chip):
        imagePath = self.getCurrentImagePath()
        filt = aosTeleState.phosimFilterID[self.band]
        fname = 'lsst_e_{}_f{}_{}_E000.fits'.format(self.obsId, filt, chip)
        img = fits.open(os.path.join(imagePath, fname))[0].data
        return img

    def getCrop(self, chip, pixX, pixY, widthPix=128):
        img = self.getImage(chip)
        # Eventually need to handle edge case.
        x = slice(pixX - widthPix // 2, pixX + widthPix // 2)
        y = slice(pixY - widthPix // 2, pixY + widthPix // 2)
        crop = img[y, x]
        return crop

    @staticmethod
    def rotateByChip(chip, image):
        raft = chip.split('_')[0]

        raftToRotations = {
            'R00': 0,
            'R04': 1,
            'R44': 2,
            'R40': 3
        }

        image = np.rot90(image, raftToRotations[raft])
        return image


    def checkZ4C(self, state, metr, debugLevel):
        z4c = np.loadtxt(self.zFile)  # in micron
        # z4cE001 = np.loadtxt(self.zFile[1])
        z4cTrue = np.zeros((metr.nFieldp4, self.znwcs, state.nOPDw))
        aa = np.loadtxt(state.zTrueFile)
        for i in range(state.nOPDw):
            z4cTrue[:, :, i] = aa[i*metr.nFieldp4:(i+1)*metr.nFieldp4, :]

        x = range(4, self.znwcs + 1)
        plt.figure(figsize=(10, 8))
        if state.inst[:4] == 'lsst':
            # subplots go like this
            #  2 1
            #  3 4
            pIdx = [2, 1, 3, 4]
            nRow = 2
            nCol = 2
        elif state.inst[:6] == 'comcam':
            pIdx = [7, 4, 1, 8, 5, 2, 9, 6, 3]
            nRow = 3
            nCol = 3

        for i in range(self.nWFS):
            chipStr, px, py = state.fieldXY2Chip(
                metr.fieldXp[i + metr.nFieldp4 - self.nWFS],
                metr.fieldYp[i + metr.nFieldp4 - self.nWFS], debugLevel)
            plt.subplot(nRow, nCol, pIdx[i])
            plt.plot(x, z4c[i, :self.znwcs3], label='CWFS_E000',
                     marker='*', color='r', markersize=6)
            # plt.plot(x, z4cE001[i, :self.znwcs3], label='CWFS_E001',
            #          marker='v', color='g', markersize=6)
            for irun in range(state.nOPDw):
                if irun==0:
                    mylabel = 'Truth'
                else:
                    mylabel = ''
                plt.plot(x, z4cTrue[i + metr.nFieldp4 - self.nWFS, 3:self.znwcs,
                                        irun],
                             label=mylabel,
                        marker='.', color='b', markersize=10)
            if ((state.inst[:4] == 'lsst' and (i == 1 or i == 2)) or
                    (state.inst[:6] == 'comcam' and (i <= 2))):
                plt.ylabel('$\mu$m')
            if ((state.inst[:4] == 'lsst' and (i == 2 or i == 3)) or
                    (state.inst[:6] == 'comcam' and (i % nRow == 0))):
                plt.xlabel('Zernike Index')
            leg = plt.legend(loc="best")
            leg.get_frame().set_alpha(0.5)
            plt.grid()
            plt.title('Zernikes %s' % chipStr, fontsize=10)

        plt.savefig(self.zCompFile, bbox_inches='tight')

    def getZ4CfromBase(self, baserun, state):
        if not os.path.isfile(self.zFile):
            baseFile = self.zFile.replace(
                'sim%d' % state.iSim, 'sim%d' % baserun)
            os.link(baseFile, self.zFile)
        if not os.path.isfile(self.zCompFile):
            baseFile = self.zCompFile.replace(
                'sim%d' % state.iSim, 'sim%d' % baserun)
            os.link(baseFile, self.zCompFile)