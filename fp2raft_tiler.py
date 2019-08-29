'''
This script divides copies of the focal plane into individual rafts
This script is designed to be called from the command line as: 

python hp2fp_tiler.py [fpID] [chunkSize]

- fpID refers to the index of the focal plane we are writing in the list stored in utils/pointingList.obj
- chunkSize is not required, and is used to reduce memory usage. The lower chunkSize is, the less memory will
be used, but the process will also be slower

The new file created stores: RA, Dec, Half Light Radius ('SIZE'), Ellipticity ('EPSILON'), and magnitude ('LMAG').
It also stores the pointing position of this "telescope" in the 'TRA' and 'TDEC' slots.
'''

import numpy as np
import os
import lsst.geom as geom
from lsst.obs.lsst import LsstCamMapper as camMapper
from lsst.obs.lsst.lsstCamMapper import getWcsFromDetector
import fitsio
from fitsio import FITS
import sys
import pickle

if len(sys.argv) == 1:
    raise(Exception('Must supply a fpIndex'))
elif len(sys.argv) == 2:
    fpID = int(sys.argv[1])
    chunkSize = 100000
elif len(sys.argv) == 3:
    fpID= int(sys.argv[1])
    chunkSize = int(sys.argv[2])
else:
    raise(Exception('Too many arguments'))

### Utility functions
def getRaftNo(ra, dec, wcsList):
    loc = geom.SpherePoint(ra, dec, geom.degrees)
    for det, wcs in wcsList.items():
        dPix = wcs.skyToPixel(loc)
        pix = geom.Point2I(dPix)
        if det.getBBox().contains(pix):
            return det.getId() // 9
    return -1

# Get pointing list from file
with open('utils/pointingList.obj', 'rb') as pl:
    pointingList = pickle.load(pl)

### Write focal plane copies to fits files

inRoot = '/nfs/slac/g/ki/ki19/lsst/jrovee/outputs/fpCopies'
fNamePattern = 'fpCopy{}.fits'
fpFile = os.path.join(inRoot,fNamePattern.format(fpID))

camera = camMapper._makeCamera()
boresight = pointingList[fpID]
wcsList = {detector : getWcsFromDetector(detector, boresight) for detector in camera}
det2raft = lambda det : det.getId() // 9

outRoot = '/nfs/slac/g/ki/ki19/lsst/jrovee/outputs/raftCopies'
outFiles = []
for raftNo in range(21):
    fname = os.path.join(outRoot, 'fpCopy{}'.format(fpID), 'raft{}.fits'.format(raftNo))
    outFiles.append(FITS(fname, 'rw', clobber=True))
writtenIn = np.zeros(21, dtype=bool)

length = len(fitsio.read(fpFile, columns=[], ext=1))
nChunks = -(-length // chunkSize)  # Ceiling integer division
for i in range(nChunks): 
    print(i)
    if i != nChunks - 1:
        span = range(chunkSize*i, chunkSize*(i+1))
    else: # We treat the last chunk slightly different because it is a different size
        span = range(chunkSize*i, length)
    ra = fitsio.read(fpFile, columns='RA', rows=span, ext=1)
    dec = fitsio.read(fpFile, columns='DEC', rows=span, ext=1)
    usefulRows = [[] for _ in range(21)]
    for j, k in enumerate(span):
        usefulRows[getRaftNo(ra[j],dec[j],wcsList)].append(k) 
    for raftNo in range(21):
        if usefulRows[raftNo]:
            data = fitsio.read(fpFile, rows=usefulRows[raftNo], columns=['RA', 'DEC', 'SIZE', 'EPSILON', 'LMAG', 'TRA', 'TDEC'], ext=1)
            data['TRA'] = boresight.getRa().asDegrees()
            data['TDEC'] = boresight.getDec().asDegrees()
            if writtenIn[raftNo]:
                outFiles[raftNo][1].append(data)
            else:
                outFiles[raftNo].write(data)
                writtenIn[raftNo] = True
