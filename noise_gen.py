import numpy as np

class Noise:
	'''
	A class that contains methods for generating noise for the entire FP.

	Each method returns a nested dictionary whose first key is the CCD ID,
	whose second key is the amplifier number, and whose values are ndarrays
	containing a noise value (in ADU) for each pixel on that amplifier.
	
	Return format is: ffp_noise['R##_S##'][amp#] = np.array
	'''
	def __init__(self):
		self.rafts = ['R{}{}'.format(i,j) for i in range(5) for j in range(5) if i in range(1,4) or j in range(1,4)]
		self.slots = ['S{}{}'.format(i//3, i%3) for i in range(9)]
		self.CCD_list = ['{}_{}'.format(raft,slot) for raft in self.rafts for slot in self.slots]
		self.ffp_noise = {}			# Stores the values of the noise for this noise object
		for CCD in self.CCD_list:
			self.ffp_noise[CCD] = {} 
		#self.e2v_rafts = []
		#self.sta_rafts = []

	def __getImagingShape__(self, ccdid):
		'''Returns the shape of the imaging section of the CCD'''
		'''
		if raft[:3] in self.e2v_rafts:
			return (2002,512)
		if raft[:3] in self.sta_rafts:
			return (2000,509)
		'''
		return (2000,509)

	def setZero(self):
		'''Returns zero noise'''
		for CCD in self.CCD_list:
			for ampno in range(1,17):
				self.ffp_noise[CCD][ampno] = np.zeros(self.__getImagingShape__(CCD))

	def setIndNoise(self, sigma):
		'''Sets the ffp_noise to random noise with std sigma and with each pixel independent'''
		for CCD in self.CCD_list:
			for ampno in range(1,17):
				self.ffp_noise[CCD][ampno] = np.random.normal(0,sigma,size=self.__getImagingShape__(CCD))

	def setCCDCorrNoise(self, cov):
		'''Takes as input a 16x16 covariance matrix for a single CCD and sets noise for the 
		FP where CCDs are assumed to be independent of each other and normally distributed 
		within the CCD'''
		for CCD in self.CCD_list:
			noise = np.random.multivariate_normal(np.zeros(16), cov,size=self.__getImagingShape__(CCD))
			for ampno in range(1,17):
				self.ffp_noise[CCD][ampno] = noise[:,:,ampno-1]

	def setMultiCCDCorrNoise(self, covStack):
		'''Takes as input a 9x16x16 array of covariance matrices for the nine CCDs on a raft
		and sets the noise for the FP where the CCDs are assumed to be independent of each other 
		and normally distributed within the CCD. Randomly shuffles order of covariance matrices
		on the raft'''
		for raft in self.rafts:
			np.random.shuffle(covStack)
			for islot, slot in enumerate(self.slots):
				ccdid = '{}_{}'.format(raft,slot)
				size = self.__getImagingShape__(ccdid)
				noise = np.random.multivariate_normal(np.zeros(16),covStack[islot,:,:], size=size)
				for ampno in range(1,17):
					self.ffp_noise[ccdid][ampno] = noise[:,:,ampno-1]

	def setRaftCorrNoise(self, cov):
		'''Takes as input a 144x144 covariance matrix for an entire raft, where the index is equal 
		to slot# * 16 + amp# - 1 and sets the noise for the FP where the rafts are assumed to be 
		independent of each other'''
		for raft in self.rafts:
			size = self.__getImagingShape__('{}_S00'.format(raft))
			noise = np.random.multivariate_normal(np.zeros(16),covStack[islot,:,:], size=size)
			for islot, slot in enumerate(self.slots):
				ccdid = '{}_{}'.format(raft,slot)
				for ampno in range(1,17):
					noiseIndex = islot * 16 + ampno - 1
					self.ffp_noise[ccdid][ampno] = noise[:,:,noiseIndex]

	def get_footprint(self, edgelen, ccdid, ampno, x, y):
		'''Given an edgelen, ccdid, ampno, and x and y coord for the bottom-left-most coord in a 
		footprint, returns an array of the noise values for that region. Assumes edgelen < 509 '''
		
		ccd_size = self.__getImagingShape__(ccdid)
		if edgelen + x < ccd_size[0] and edgelen + y < ccd_size[1]:
			return self.ffp_noise[ccdid][ampno][x:x+edgelen,y:y+edgelen]

	def get_ffp_noise(self):
		'''Returns the noise for the entire focal plane'''
		return self.ffp_noise
