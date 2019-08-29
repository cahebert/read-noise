'''
This script divides Joe Derose's Chinchilla catalogs into geometries that match the LSST focal plane geometry. 
This script is designed to be called from the command line as: 

python hp2fp_tiler.py [fpID] [chunkSize]

- fpID refers to the index of the focal plane we are writing in the list stored in utils/pointingList.obj
- chunkSize is not required, and is used to reduce memory usage. The lower chunkSize is, the less memory will
be used, but the process will also be slower

The new file created stores: RA, Dec, Half Light Radius ('SIZE'), Ellipticity ('EPSILON'), and magnitude ('LMAG').
It also stores the pointing position of this "telescope" in the 'TRA' and 'TDEC' slots.

Julian Rovee
'''
import numpy as np
import os
import glob
import healpy as hp
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
    chunkSize = 10000
elif len(sys.argv) == 3:
    fpID = int(sys.argv[1])
    chunkSize = int(sys.argv[2])
else:
    raise(Exception('Too many arguments'))

### Utility functions
def inBoundsFast(ra, dec):
        loc = geom.SpherePoint(ra, dec, geom.degrees)
        distance = loc.separation(boresight)
        if distance.asDegrees() > 2.2:
            return False
        for det in wcsList:
            pix = geom.Point2I(wcsList[det].skyToPixel(loc))
            if det.getBBox().contains(pix):
                return True
        return False

def getPixelsFromCenter(ra, dec, camera=None):
    '''
    Returns all pixels which overlap with the field of view of the camera pointed at position (ra, dec) on the sky
    '''
    if camera is None:
        camera = camMapper._makeCamera()

    boresight = geom.SpherePoint(ra, dec, geom.degrees)
    corners = [
        getWcsFromDetector(camera['R41_S20'], boresight).pixelToSky(0,4071),
        getWcsFromDetector(camera['R43_S22'], boresight).pixelToSky(3999,4071),
        getWcsFromDetector(camera['R34_S22'], boresight).pixelToSky(3999,4071),
        getWcsFromDetector(camera['R14_S02'], boresight).pixelToSky(3999,0),
        getWcsFromDetector(camera['R03_S02'], boresight).pixelToSky(3999,0),
        getWcsFromDetector(camera['R01_S00'], boresight).pixelToSky(0,0),
        getWcsFromDetector(camera['R10_S00'], boresight).pixelToSky(0,0),
        getWcsFromDetector(camera['R30_S20'], boresight).pixelToSky(0,4071)
    ]
    cornerRAList = np.array([corner.getRa().asDegrees() for corner in corners])
    cornerDecList = np.array([corner.getDec().asDegrees() for corner in corners])
    vecList = hp.ang2vec(cornerRAList,cornerDecList,lonlat=True)
    return hp.query_polygon(8,vecList,inclusive=True,nest=True, fact = 128)

### Write focal plane copies to fits files

root = "/nfs/slac/des/fs1/g/sims/jderose/BCC/Chinchilla/Herd/Chinchilla-4/v1.9.2/addgalspostprocess/truth/truth"
fNamePattern = "Chinchilla-4_lensed.{}.fits"

camera = camMapper._makeCamera()
with open('/u/gu/jrovee//WORK/read-noise/utils/pointingList.obj', 'rb') as pointingList:
    boresight = pickle.load(pointingList)[fpID]
pixels = getPixelsFromCenter(boresight.getRa().asDegrees(), boresight.getDec().asDegrees(), camera)
wcsList = {detector : getWcsFromDetector(detector, boresight) for detector in camera}
testFiles = [os.path.join(root,fNamePattern.format(pixel)) for pixel in pixels]

outFile = '/nfs/slac/g/ki/ki19/lsst/jrovee/outputs/fpCopies/fpCopy{}.fits'.format(fpID)
with FITS(outFile, 'rw', clobber=True) as fits:
    writtenIn = False
    for file in testFiles:
        length = len(fitsio.read(file, columns=[], ext=1))
        nChunks = -(-length // chunkSize)  # Ceiling integer division
        for i in range(nChunks): 
            if i != nChunks - 1:
                span = range(chunkSize*i, chunkSize*(i+1))
            else: # We treat the last chunk slightly different because it is a different size
                span = range(chunkSize*i, length)
            ra = fitsio.read(file, columns='RA', rows=span, ext=1)
            dec = fitsio.read(file, columns='DEC', rows=span, ext=1)
            usefulRows = [k for j, k in enumerate(span) if inBoundsFast(ra[j],dec[j])]
            if usefulRows:
                data = fitsio.read(file, rows=usefulRows, columns=['RA', 'DEC', 'SIZE', 'EPSILON', 'LMAG', 'TRA', 'TDEC'], ext=1)
                data['TRA'] = boresight.getRa().asDegrees()
                data['TDEC'] = boresight.getDec().asDegrees()
                if writtenIn:
                    fits[1].append(data)
                else:
                    fits.write(data)
                    writtenIn = True
