from noise_gen import Noise
import numpy as np
import fitsio
import sys
import lsst.afw.geom as geom
from lsst.obs.lsst import lsstCamMapper as camMapper
from scipy.stats import multivariate_normal

fname = 'fpCopy%s.fits' % sys.argv[1]

pointingRA = fitsio.read(fname, columns='TRA', rows=0, ext=1)[0]
pointingDec = fitsio.read(fname, columns='TDEC', rows=0, ext=1)[0]
boresight = geom.SpherePoint(pointingRA, pointingDec, geom.degrees)
wcsList = {detector : getWcsFromDetector(detector, boresight) for detector in camera}

def skyToCamPixel(ra, dec):
    loc = geom.SpherePoint(ra, dec, geom.degrees)
    for det in wcsList:
        pix = geom.Point2I(wcsList[det].skyToPixel(loc))
        if det.getBBox().contains(pix):
            return det.getName(), pix.getX(), pix.getY()
    return 'OOB', 0, 0

camera = camMapper._makeCamera()
noise = Noise(camera)

noise_type = sys.argv[2]
if noise_type == 'NONE':
	noise.setZeroNoise()
if noise_type == 'IND':
	sigma = int(sys.argv[3])
	noise.setIndNoise(sigma)
if noise_type == 'CCD':
	corr_matrix = np.load(sys.argv[3])
	noise.setCCDCorrNoise(corr_matrix)
if noise_type == 'MULTICCD':
	corr_matrices = np.load(sys.argv[3])
	noise.setMultiCCDCorrNoise(corr_matrices)
if noise_type == 'RAFT':
	corr_matrix = np.load(sys.argv[3])
	noise.setRaftCorrNoise(corr_matrix)

outFile = 'fpMod{}_{}.fits'.format(sys.argv[1], sys.argv[2])

def getShearMat(e1, e2):
	absesq = e1**2 + e2**2
	if absesq > 1.e-4:
		e2g = 1. / (1. + np.sqrt(1.-absesq))
	else:
		e2g = 0.5 + absesq*(0.125 + absesq*(0.0625 + absesq*0.0390625))
	g1 = e1 / e2g
	g2 = e2 / e2g
	return np.array([[ 1 + g1 ,   g2   ], 
					 [   g2   , 1 - g1 ]])

def getPsfMat(sigma):
	return np.array([[sigma**2, 0], 
		             [0, sigma**2]])

def getWeightsAndSize(e1, e2, hlr, psfSig=0.7):
	galSig = hlr / 1.1774100225154747
	cov = (getShearMat(e1, e2) + getPsfMat(psfSig)) * galSig**2 / 0.2**2
	
	gridSize= 8 * np.sqrt(galSig**2 + psfSig**2)
	x, y = np.mgrid[-1*(gridSize - 1)/2:(gridSize + 1)/2:1, -1*(gridSize - 1)/2:(gridSize + 1)/2:1]
	pos = np.zeros(x.shape + (2,))
	pos[:, :, 0] = x; pos[:, :, 1] = y

	return multivariate_normal.pdf(pos, cov=cov), gridSize 

def getNoise(ra, dec, edgelen):
	ccdid, cx, cy = skyToCamPixel(ra, dec)
	return noise.getFootprint(ccdid, cx, cy, edgelen) * 0.0009510798216162556

def getDeltaFlux(weightMat, noiseMat):
	return np.sum(np.multiply(weightMat, noiseMat)) / np.sum(np.multiply(weightMat, weightMat))

nElems = len(fitsio.read(file, columns=[], ext=1))

for i in range(nElems):
	weights, edgelen = getWeightsAndSize(E1[i], E2[i], fits['HLR'][i])
	noise = getNoise(fits['RA'][i], fits['DEC'][i], edgelen)
	getDeltaFlux(weights, noise)

