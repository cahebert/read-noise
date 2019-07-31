import numpy as np
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms

class Noise:
	'''
	A class that contains methods for generating noise for the entire FP.

	Each method returns a nested dictionary whose first key is the CCD ID,
	whose second key is the amplifier number, and whose values are ndarrays
	containing a noise value (in ADU) for each pixel on that amplifier.
	
	Return format is: ffp_noise['R##_S##'][amp#] = np.array
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
		'''Returns the shape of the imaging section of one amplifier on the ccd'''
		
		if ccdid in self.e2v_detectors:
			return 512,2002
		if ccdid in self.sta_detectors:
			return 509,2000
		raise Exception('Not a valid ccdid')

	def getCCDSize(self,ccdid):
		ampShapeX, ampShapeY = self.getImagingShape(ccdid)
		return ampShapeX * 8, ampShapeY * 2

	def getCCDfromAmps(self, CCD, noise, correction=True):
		'''Takes a dictionary (ampno : amplifier noise) and returns the noise in CCD coordinates'''

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
		ffp_noise = {}
		for CCD in self.CCD_list:
			ffp_noise[CCD] = {}
			for ampno in range(1,17):
				ffp_noise[CCD][ampno] = np.zeros(self.getImagingShape(CCD))
			self.CCD_noise = self.getCCDfromAmps(CCD, ffp_noise[CCD])

	def setIndNoise(self, sigma):
		'''Sets the ffp_noise to random noise with std sigma and with each pixel independent'''
		ffp_noise = {}
		for CCD in self.CCD_list:
			ffp_noise[CCD] = {}
			for ampno in range(1,17):
				ffp_noise[CCD][ampno] = np.random.normal(0,sigma,size=self.getImagingShape(CCD))
			self.CCD_noise = self.getCCDfromAmps(CCD, ffp_noise[CCD])


	def setCCDCorrNoise(self, cov):
		'''Takes as input a 16x16 covariance matrix for a single CCD and sets noise for the 
		FP where CCDs are assumed to be independent of each other and normally distributed 
		within the CCD'''
		ffp_noise = {}
		for CCD in self.CCD_list:
			ffp_noise[CCD] = {}
			noise = np.random.multivariate_normal(np.zeros(16), cov,size=self.getImagingShape(CCD))
			for ampno in range(1,17):
				ffp_noise[CCD][ampno] = noise[:,:,ampno-1]
			self.CCD_noise = self.getCCDfromAmps(CCD, ffp_noise[CCD])

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
				self.CCD_noise = self.getCCDfromAmps(ccdid, ffp_noise[ccdid])

	def setRaftCorrNoise(self, cov):
		'''Takes as input a 144x144 covariance matrix for an entire raft, where the index is equal 
		to slot# * 16 + amp# - 1 and sets the noise for the FP where the rafts are assumed to be 
		independent of each other'''
		ffp_noise = {}
		for raft in self.rafts:
			size = self.getImagingShape('{}_S00'.format(raft))
			noise = np.random.multivariate_normal(np.zeros(16),covStack[islot,:,:], size=size)
			for islot, slot in enumerate(self.slots):
				ccdid = '{}_{}'.format(raft,slot)

				for ampno in range(1,17):
					noiseIndex = islot * 16 + ampno - 1
					ffp_noise[ccdid][ampno] = noise[:,:,noiseIndex]
				self.CCD_noise = self.getCCDfromAmps(ccdid, ffp_noise[ccdid])

	@staticmethod
	def get_adj_ccd(currCCD, xDir=None, yDir=None, checkValid=False):
		'''Given a ccdid, it will return the ccdid of the CCD in the direction 'direction' which is one 
		of 'UP', 'DOWN', 'LEFT', or 'RIGHT' '''

		ccdX = 3 * int(currCCD[1]) + int(currCCD[-2])
		ccdY = 3 * int(currCCD[2]) + int(currCCD[-1])

		if yDir == 'UP':
			ccdY += 1
		elif yDir == 'DOWN':
			ccdY -= 1
		if xDir == 'LEFT':
			ccdX += 1
		elif xDir == 'RIGHT':
			ccdX -= 1
		

		raftX = ccdX//3
		raftY = ccdY//3
		if checkValid:
			if (raftX, raftY) in {(0,0),(4,4),(0,4),(4,0)} or raftX not in range(5) or ccdY not in range(5):
				raise Exception('No CCD in that direction')

		return 'R{}{}_S{}{}'.format(raftX, raftY, ccdX%3, ccdY%3)
	"""
	@classmethod
	def get_footprint_from_vertices(self, ccdid, cx, cy, edgelen):
		'''Given a value for the center of an object, returns a square array with
		edgelength edgelen containing the noise centered at pixel (cx, cy) on the ccdid '''
		halfEdge = edgelen / 2
		size = self.getCCDSize(ccdid)
		bottomBound = cx - halfEdge + 0.5
		topBound = cx + halfEdge + 0.5
		leftBound = cy - halfEdge + 0.5
		rightBound = cy + halfEdge + 0.5

		footprint = np.zeros((edgelen, edgelen))

		lb2 = leftBound % size[1]
		rb2 = rightBound % size[1]
		tb2 = topBound % size[0]
		bb2 = bottomBound % size[0]



		if leftBound < 0:
			lrDiv = -1 * leftBound
			ccdL = self.get_adj_ccd(ccdid, xDir = 'LEFT')
			if bottomBound < 0:
				udDiv = -1 * bottomBound
				ccdD = self.get_adj_ccd(ccdid, yDir = 'DOWN')
				ccdLD = self.get_adj_ccd(ccdid, xDir = 'LEFT', yDir = 'DOWN')
				footprint[udDiv:,lrDiv:] = self.CCD_noise[ccdid][:topBound,:rightBound]
				footprint[udDiv:,:lrDiv] = self.CCD_noise[ccdL][:topBound,leftBound:]
				footprint[:udDiv,lrDiv:] = self.CCD_noise[ccdD][bottomBound:,:rightBound]
				footprint[:udDiv,:lrDiv] = self.CCD_noise[ccdLD][bottomBound:,leftBound:]
			elif topBound > size[0]:
				udDiv = size[0] - bottomBound
				ccdU = self.get_adj_ccd(ccdid, yDir = 'UP')
				ccdLU = self.get_adj_ccd(ccdid, xDir = 'LEFT', yDir = 'UP')
				footprint[udDiv:,lrDiv:] = self.CCD_noise[ccdLU][:topBound,:rightBound]
				footprint[udDiv:,:lrDiv] = self.CCD_noise[ccdU][:topBound,leftBound:]
				footprint[:udDiv,lrDiv:] = self.CCD_noise[ccdL][bottomBound:,:rightBound]
				footprint[:udDiv,:lrDiv] = self.CCD_noise[ccdid][bottomBound:,leftBound:]
			else:
				footprint[:,:lrDiv] = self.CCD_noise[ccdL][bottomBound:topBound,leftBound:]
				footprint[:,lrDiv:] = self.CCD_noise[ccdid][bottomBound:topBound,:rightBound]
		elif rightBound > size[1]:
			lrDiv = size[1] - leftBound
			ccdR = self.get_adj_ccd(ccdid, xDir = 'RIGHT')
			if bottomBound < 0:
				footprint[:,:] = self.CCD_noise[ccdid][]
			elif topBound > size[0]:
				footprint[:,:] = self.CCD_noise[ccdid][]
			else:
				footprint[:,:lrDiv] = self.CCD_noise[ccdid][bottomBound:topBound,leftBound:]
				footprint[:,lrDiv:] = self.CCD_noise[ccdR][bottomBound:topBound,:rightBound-size[1]]
		else:
			if bottomBound < 0:
				footprint[:,:] = self.CCD_noise[ccdid][]
			elif topBound > size[0]:
				footprint[:,:] = self.CCD_noise[ccdid][]
			else:
				footprint = self.CCD_noise[ccdid][leftBound:rightBound, bottomBound:topBound]
	"""

	def getFootprint(self, ccdid, cx, cy, edgelen):
		'''Given a value for the center of an object, returns a square array with
		edgelength edgelen containing the noise centered at pixel (cx, cy) on the ccdid '''
		halfEdge = edgelen / 2
		size = self.getCCDSize(ccdid)
		bottomBound = int(cx - halfEdge + 0.5)
		topBound = int(cx + halfEdge + 0.5)
		leftBound = int(cy - halfEdge + 0.5)
		rightBound = int(cy + halfEdge + 0.5)

		if bottomBound < 0 or topBound > size[0] or leftBound < 0 or rightBound > size[1]:
			return None

		return self.CCD_noise[ccdid][bottomBound:topBound,leftBound:rightBound]	
