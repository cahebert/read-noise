'''
Takes 3 additional arguments:

1: the ID of the focal plane copy we are manipulating
2: the type of noise we are adding. Must be one of {'NONE', 'IND', 'CCD', 'MULTICCD', 'RAFT'}
3: additional specifications for certain types of noise.

'''

from noise_gen import Noise, BoundaryError
import numpy as np
import fitsio
import sys
import lsst.afw.geom as geom
from lsst.obs.lsst import LsstCamMapper as camMapper
from lsst.obs.lsst.lsstCamMapper import getWcsFromDetector
from scipy.stats import multivariate_normal

fpID = sys.argv[1]
inFile = '/nfs/slac/g/ki/ki19/lsst/jrovee/outputs/fpCopy%s.fits' % fpID

### Establish camera and wcs
camera = camMapper._makeCamera()
pointingRA = fitsio.read(inFile, columns='TRA', rows=0, ext=1)[0]
pointingDec = fitsio.read(inFile, columns='TDEC', rows=0, ext=1)[0]
boresight = geom.SpherePoint(pointingRA, pointingDec, geom.degrees)
wcsList = {detector : getWcsFromDetector(detector, boresight) for detector in camera}

def skyToCamPixel(ra, dec):
    loc = geom.SpherePoint(ra, dec, geom.degrees)
    for det in wcsList:
        pix = geom.Point2I(wcsList[det].skyToPixel(loc))
        if det.getBBox().contains(pix):
            return det.getName(), int(pix.getX()), int(pix.getY())
    return 'OOB', 0, 0

### Make noise
noise = Noise(camera)
noise_type = sys.argv[2]

# Adds no noise. Do not supply a third argument 
if noise_type == 'NONE':   
	noise.setZeroNoise()

# Adds uncorrelated noise distributed as a gaussian for each pixel.
# Third argument is the sigma for the noise (in ADU)
if noise_type == 'IND':
	sigma = int(sys.argv[3])
	noise.setIndNoise(sigma)

# Adds noise correlated between the amplifiers within a single chip.
# Third argument is the filename for the 16x16 covariance matrix being used
if noise_type == 'CCD':
	corr_matrix = np.load(sys.argv[3])
	noise.setCCDCorrNoise(corr_matrix)

# Adds noise correlated between the amplifiers within a single chip, but lets
# you supply 9 covariance matrices which will be chosen randomly for each raft.
# Third argument is the filename for the 9x16x16 stack of covariance matrices being used
if noise_type == 'MULTICCD':
	corr_matrices = np.load(sys.argv[3])
	noise.setMultiCCDCorrNoise(corr_matrices)

# Adds noise correlated between all amplifiers on the same raft. Third argument is 
# the filename for the 144x144 covariance matrix being used.
if noise_type == 'RAFT':
	corr_matrix = np.load(sys.argv[3])
	noise.setRaftCorrNoise(corr_matrix)

### Emulation functions

def getShearMat(e1, e2):
	'''returns the shear matrix for the given e1, e2, inspired by GalSim'''
	absesq = e1**2 + e2**2
	if absesq > 1.e-4:
		e2g = 1. / (1. + np.sqrt(1.-absesq))
	else:
		e2g = 0.5 + absesq*(0.125 + absesq*(0.0625 + absesq*0.0390625))
	g1 = e1 * e2g
	g2 = e2 * e2g
	return np.array([ [ 1 + g1 , g2 ], [ g2 , 1 - g1 ] ])

def getPsfMat(sigma):
	'''Returns a PSF correlation matrix for a given sigma'''
	return np.array([ [ sigma**2 , 0 ], [ 0 , sigma**2 ] ])

def getGridSize(hlr, psfSig=0.7):
	'''Returns the side length of the postage stamp we are using to model the galaxy. Approximately 
	3 sigma in each direction, where 'sigma' is a combined measure of the galaxy and psf sigmas'''
	galSig = hlr / 1.1774100225154747
	return int(6 * np.sqrt(galSig**2 + psfSig**2) / 0.2)
	
def getWeights(e1, e2, hlr, psfSig=0.7):
	'''
	Gets a weighting scheme that assumes a gaussian profile for a galaxy with given e1, e2, and hlr. 
	Also returns the side length
	'''
	galSig = hlr / 1.1774100225154747 # Divide by sqrt(ln(2)) to convert from hlr to sigma
	cov = (getShearMat(e1, e2) * galSig**2 + getPsfMat(psfSig)) / 0.2**2

	gridSize = getGridSize(hlr, psfSig)
	x, y = np.mgrid[-1*(gridSize - 1)/2:(gridSize + 1)/2:1, -1*(gridSize - 1)/2:(gridSize + 1)/2:1]
	pos = np.zeros(x.shape + (2,))
	pos[:, :, 0] = x; pos[:, :, 1] = y

	return multivariate_normal.pdf(pos, cov=cov)

def getNoise(ra, dec, edgelen):
	'''Gets the noise at a particular position in nanomaggies.'''
	ccdid, cx, cy = skyToCamPixel(ra, dec)
	return noise.getFootprint(ccdid, cx, cy, edgelen) * 0.0009510798216162556 # ADUs to nanomaggies conversion

def getDeltaFlux(weightMat, noiseMat):
	'''Calculates a weighted sum of noise in noiseMat using the weightMat as a weighting scheme'''
	return np.sum(np.multiply(weightMat, noiseMat)) / np.sum(np.multiply(weightMat, weightMat))

# Process fits
chunkSize = sys.argv[-1] 
length = len(fitsio.read(inFile, columns=[], ext=1))
nChunks = -(-length // chunkSize)  # Cieling integer division

outFile = '/nfs/slac/g/ki/ki19/lsst/jrovee/modified/fpMod{}_{}.fits'.format(fpID, noise_type)
maskFile = '/nfs/slac/g/ki/ki19/lsst/jrovee/modified/mask{}_{}.npy'.format(fpID, noise_type)

nElems = len(fitsio.read(inFile, columns=[], ext=1))
newRband = np.zeros(nElems)
mask = np.ones(nElems, dtype=bool)

with FITS(outFile, 'rw', clobber=True) as fits:
	writtenIn = False
	for i in range(nChunks): 
		if i != nChunks - 1:
			span = range(chunkSize*i, chunkSize*(i+1))
		else: 
			span = range(chunkSize*i, length)
		outData = fitsio.read(inFile, columns=['RA', 'DEC', 'LMAG'], rows=span, ext=1)
		rFlux = 10**((outData['LMAG'][:,1]-22.5)/(-2.5)) # DE rband
		hlr = fitsio.read(inFile, columns='SIZE', rows=span, ext = 1)
		E1 = fitsio.read(inFile, columns='EPSILON', rows=span, ext = 1)[:,0]
		E2 = fitsio.read(inFile, columns='EPSILON', rows=span, ext = 1)[:,1]

		newRFlux = np.zeros(chunkSize)
		for j, k in enumerate(span):
			try:
				noiseFootprint = getNoise(outData['RA'][j], outData['DEC'][j], getGridSize(hlr[j]))
				weights = getWeights(E1[j], E2[j], hlr[j])
				totalFlux = getDeltaFlux(weights, noiseFootprint) + rFlux[j]
				if totalFlux > 0:
					mag = 22.5 - 2.5*np.log10(totalFlux)
					outData['LMAG'][j,1] = mag
				else:
					mask[k] = False
			except BoundaryError:
				mask[k] = False
		if not writtenIn:
			fits.write(outData)
		else:
			fits.append(outData)

np.save(maskFile, mask)
