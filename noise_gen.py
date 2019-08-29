'''
This file defines two objects, FocalPlaneNoise and RaftNoise, which each help us store noise
corresponding to the LSST camera. FocalPlaneNoise stores noise values for every pixel on the
focal plane at the same time, while RaftNoise only stores noise for a single raft.

This file also defines a custom exception that may be raised in the getFootprint method of
both of the above classes.

In order to use this object:
1. Import one of the noise-generator classes. 
2. Generate the noise using one of the setter methods: setZeroNoise, setIndNoise, setCcdNoise,
   setMultiCcdNoise, setRaftNoise
3. In order to access noise, either get it through the CCD_noise class attribute or use the 
   getFootprint method to get a small 'postage stamp' subset of the noise

Julian Rovee
'''
import numpy as np
import os
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms

class BoundaryError(Exception):
	pass

class FocalPlaneNoise:
	'''
	A class that contains methods for simulating noise for the entire LSST focal plane.

	Noise is stored in the CCD_noise attribute as a dictionary which maps detector name to an
	ndarray containing the noise corresponding to that detector.

	Except for the zero noise case, all noise models are gaussian.
	'''
	def __init__(self, camera):
		self.lct = LsstCameraTransforms(camera)
		self.CCD_list = [det.getName() for det in camera]
		self.rafts = {ccd[:3] for ccd in self.CCD_list}
		self.slots = ['S{}{}'.format(i//3, i%3) for i in range(9)]
		
		self.e2v_detectors = {det.getName() for det in camera if det.getSerial().find('e2v') != -1}
		self.sta_detectors = {det.getName() for det in camera if det.getSerial().find('3800C') != -1}
		
		self.CCD_noise = {}
		
	def getImagingShape(self, ccdid):
		if ccdid in self.e2v_detectors:
			return 512,2002
		if ccdid in self.sta_detectors:
			return 509,2000
		raise Exception('Not a valid ccdid')

	def getCCDSize(self,ccdid):
		ampShapeX, ampShapeY = self.getImagingShape(ccdid)
		return ampShapeX * 8, ampShapeY * 2

	def getCCDfromAmps(self, CCD, noise, correction=True):
		'''Takes a dictionary (ampno : amplifier noise) and returns an ndarray 
		containing the noise in CCD coordinates'''

		xShape, yShape = self.getImagingShape(CCD)
		out = np.zeros(self.getCCDSize(CCD))
		for ampno in range(1,17):
			ampOrigX, ampOrigY = self.lct.ampPixelToCcdPixel(0,0,ampno,detectorName=CCD)
			ampCornerX, ampCornerY = self.lct.ampPixelToCcdPixel(xShape-1, yShape-1,ampno,detectorName=CCD)
			if correction:
				ampCornerX -= 3
			if ampOrigX > ampCornerX:
				noise[ampno] = np.flipud(noise[ampno])
			if ampOrigY > ampCornerY:
				noise[ampno] = np.fliplr(noise[ampno])
			bottomLeftX = min(ampCornerX,ampOrigX)
			bottomLeftY = min(ampCornerY,ampOrigY)
			out[bottomLeftX:bottomLeftX+xShape, bottomLeftY:bottomLeftY+yShape] = noise[ampno]
		return out

	def setZeroNoise(self):
		'''Sets all noise to zero arrays'''
		for CCD in self.CCD_list:
			self.CCD_noise[CCD] = np.zeros(self.getCCDSize(CCD))

	def setIndNoise(self, sigma):
		'''Sets CCD_noise to random noise with std sigma and with each pixel independent'''
		for CCD in self.CCD_list:
			self.CCD_noise[CCD] = np.random.normal(0, sigma, size=self.getCCDSize(CCD))

	def setCCDCorrNoise(self, cov):
		'''Takes as input a 16x16 covariance matrix for a single CCD and sets noise for the 
		FP where CCDs are assumed to be independent of each other and normally distributed 
		within the CCD'''
		for CCD in self.CCD_list:
			ampNoiseContainer = {}
			noise = np.random.multivariate_normal(np.zeros(16), cov,size=self.getImagingShape(CCD))
			for ampno in range(1,17):
				ampNoiseContainer[ampno] = noise[:,:,ampno-1]
			self.CCD_noise[CCD] = self.getCCDfromAmps(CCD, ampNoiseContainer)

	def setMultiCCDCorrNoise(self, covStack, shuffle=True):
		'''Takes as input a 9x16x16 array of covariance matrices for the nine CCDs on a raft
		and sets the noise for the FP where the CCDs are assumed to be independent of each other 
		and normally distributed within the CCD. Randomly shuffles order of covariance matrices
		on the raft if shuffle is True'''
		ffp_noise = {}
		for raft in self.rafts:
			if shuffle:
				np.random.shuffle(covStack)
			for islot, slot in enumerate(self.slots):
				ccdid = '{}_{}'.format(raft,slot)
				ffp_noise[ccdid] = {}
				size = self.getImagingShape(ccdid)
				noise = np.random.multivariate_normal(np.zeros(16),covStack[islot,:,:], size=size)
				for ampno in range(1,17):
					ffp_noise[ccdid][ampno] = noise[:,:,ampno-1]
				self.CCD_noise[ccdid] = self.getCCDfromAmps(ccdid, ffp_noise[ccdid])

	def setRaftCorrNoise(self, cov):
		'''Takes as input a 144x144 covariance matrix for an entire raft, where the index is equal 
		to slot# * 16 + amp# - 1 and sets the noise for the FP where the rafts are assumed to be 
		independent of each other'''
		ffp_noise = {}
		for raft in self.rafts:
			size = self.getImagingShape('{}_S00'.format(raft))
			noise = np.random.multivariate_normal(np.zeros(16), cov, size=size)
			for islot, slot in enumerate(self.slots):
				ccdid = '{}_{}'.format(raft,slot)

				for ampno in range(1,17):
					noiseIndex = islot * 16 + ampno - 1
					ffp_noise[ccdid][ampno] = noise[:,:,noiseIndex]
				self.CCD_noise[ccdid] = self.getCCDfromAmps(ccdid, ffp_noise[ccdid])

	def getFootprint(self, ccdid, cx, cy, edgelen):
		'''Given a value for the center pixel of an object, returns a square array with
		edgelength edgelen containing the noise centered at pixel (cx, cy) on the ccdid '''
		halfEdge = edgelen / 2
		size = self.getCCDSize(ccdid)
		bottomBound = cx - halfEdge + 0.5
		topBound = cx + halfEdge + 0.5
		leftBound = cy - halfEdge + 0.5
		rightBound = cy + halfEdge + 0.5

		if bottomBound < 0 or topBound > size[0] or leftBound < 0 or rightBound > size[1]:
			raise BoundaryError

		bottomBound = int(bottomBound)
		topBound = int(topBound)
		leftBound = int(leftBound)
		rightBound = int(rightBound)

		return self.CCD_noise[ccdid][bottomBound:topBound,leftBound:rightBound]
    

class RaftNoise:
	'''
        A class that contains methods for simulating noise for a single LSST raft.

        Noise is stored in the CCD_noise attribute as a dictionary which maps detector name to an
        ndarray containing the noise corresponding to that detector.

        Except for the zero noise case, all noise models are gaussian.
        '''	

	def __init__(self, camera, raftNo):
		self.lct = LsstCameraTransforms(camera)
		self.CCD_list = [camera[i].getName() for i in range(raftNo*9, (raftNo+1)*9)]
		self.slots = ['S{}{}'.format(i//3, i%3) for i in range(9)]
		
		self.e2vDetector = camera[self.CCD_list[0]].getSerial().find('e2v') != -1 # Usually false for our purposes
		
		self.CCD_noise = {}
		
	def getImagingShape(self):
		if self.e2vDetector:
			return 512,2002
		else:
			return 509,2000

	def getCCDSize(self):
		ampShapeX, ampShapeY = self.getImagingShape()
		return ampShapeX * 8, ampShapeY * 2

	def getCCDfromAmps(self, CCD, noise, correction=True):
		'''Takes a dictionary (ampno : amplifier noise) and returns the noise in CCD coordinates'''

		xShape, yShape = self.getImagingShape()
		out = np.zeros(self.getCCDSize())
		for ampno in range(1,17):
			ampOrigX, ampOrigY = self.lct.ampPixelToCcdPixel(0,0,ampno,detectorName=CCD)
			ampCornerX, ampCornerY = self.lct.ampPixelToCcdPixel(xShape-1, yShape-1,ampno,detectorName=CCD)
			if correction:
				ampCornerX -= 3
			if ampOrigX > ampCornerX:
				noise[ampno] = np.flipud(noise[ampno])
			if ampOrigY > ampCornerY:
				noise[ampno] = np.fliplr(noise[ampno])
			bottomLeftX = min(ampCornerX,ampOrigX)
			bottomLeftY = min(ampCornerY,ampOrigY)
			out[bottomLeftX:bottomLeftX+xShape, bottomLeftY:bottomLeftY+yShape] = noise[ampno]
		return out

	def setZeroNoise(self):
		for CCD in self.CCD_list:
			self.CCD_noise[CCD] = np.zeros(self.getCCDSize())

	def setIndNoise(self, sigma):
		'''Sets the CCD_noise to random noise with std sigma and with each pixel independent'''
		for CCD in self.CCD_list:
			self.CCD_noise[CCD] = np.random.normal(0, sigma, size=self.getCCDSize())

	def setCCDCorrNoise(self, cov):
		'''Takes as input a 16x16 covariance matrix for a single CCD and sets noise for the 
		FP where CCDs are assumed to be independent of each other and normally distributed 
		within the CCD'''
		for CCD in self.CCD_list:
			ampNoiseContainer = {}
			noise = np.random.multivariate_normal(np.zeros(16), cov,size=self.getImagingShape())
			for ampno in range(1,17):
				ampNoiseContainer[ampno] = noise[:,:,ampno-1]
			self.CCD_noise[CCD] = self.getCCDfromAmps(CCD, ampNoiseContainer)

	def setMultiCCDCorrNoise(self, covStack, shuffle=True):
		'''Takes as input a 9x16x16 array of covariance matrices for the nine CCDs on a raft
		and sets the noise for the FP where the CCDs are assumed to be independent of each other 
		and normally distributed within the CCD. Randomly shuffles order of covariance matrices
		on the raft if shuffle is True'''

		if shuffle:
			np.random.shuffle(covStack)
		for ccdid in self.CCD_list:
			ampNoiseContainer = {}
			size = self.getImagingShape()
			noise = np.random.multivariate_normal(np.zeros(16),covStack[islot,:,:], size=size)
			for ampno in range(1,17):
				ampNoiseContainer[ampno] = noise[:,:,ampno-1]
			self.CCD_noise[ccdid] = self.getCCDfromAmps(ccdid, ampNoiseContainer)

	def setRaftCorrNoise(self, cov):
		'''Takes as input a 144x144 covariance matrix for an entire raft, where the index is equal 
		to slot# * 16 + amp# - 1 and sets the noise for the FP where the rafts are assumed to be 
		independent of each other'''
		
		size = self.getImagingShape()
		noise = np.random.multivariate_normal(np.zeros(16), cov, size=size)
		for islot, ccdid in self.CCD_list:
			ampNoiseContainer = {}
			for ampno in range(1,17):
				noiseIndex = islot * 16 + ampno - 1
				ampNoiseContainer[ampno] = noise[:,:,noiseIndex]
			self.CCD_noise[ccdid] = self.getCCDfromAmps(ccdid, ampNoiseContainer)

	def getFootprint(self, ccdid, cx, cy, edgelen):
		'''Given a value for the center of an object, returns a square array with
		edgelength edgelen containing the noise centered at pixel (cx, cy) on the ccdid '''
		halfEdge = edgelen / 2
		size = self.getCCDSize()
		bottomBound = cx - halfEdge + 0.5
		topBound = cx + halfEdge + 0.5
		leftBound = cy - halfEdge + 0.5
		rightBound = cy + halfEdge + 0.5

		if bottomBound < 0 or topBound > size[0] or leftBound < 0 or rightBound > size[1]:
			raise BoundaryError

		bottomBound = int(bottomBound)
		topBound = int(topBound)
		leftBound = int(leftBound)
		rightBound = int(rightBound)

		return self.CCD_noise[ccdid][bottomBound:topBound,leftBound:rightBound]
    
