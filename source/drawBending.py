#!/usr/bin/env python

# @author: Bo Xin
# @      Large Synoptic Survey Telescope

# main function

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description='-----draw bending modes------')

    parser.add_argument('mirror', choices=('M1M3', 'M2'), help='M1M3 or M2')
    args = parser.parse_args()

    if args.mirror == 'M1M3':
        # bending modes
        aa = np.loadtxt('data/M1M3/M1M3_1um_156_grid.DAT')
        # nodeID = aa[:, 0]
        bx = aa[:, 1]
        by = aa[:, 2]
        bz = aa[:, 3:]
    elif args.mirror == 'M2':
        aa = np.loadtxt('data/M2/M2_1um_grid.DAT')
        bx = aa[:, 0]
        by = aa[:, 1]
        bz = aa[:, 2:]

    nB = 20
    nRow = 4
    nCol = int(np.ceil(nB / nRow))
    f, ax = plt.subplots(nRow, nCol, figsize=(15, 10))
    for iRow in range(nRow):
        for iCol in range(nCol):
            ib = iRow * nCol + iCol
            color = (bz[:, ib] - min(bz[:, ib])) / \
                (max(bz[:, ib]) - min(bz[:, ib])) * 100
            ax[iRow, iCol].scatter(bx, by, s=5, c=color,
                                   marker='.', edgecolor='none')
            ax[iRow, iCol].axis('equal')

    plt.savefig('drawBending%s.png' % args.mirror, dpi=500)
    # plt.show()

if __name__ == "__main__":
    main()
