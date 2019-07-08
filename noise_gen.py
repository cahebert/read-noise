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

	def getIndNoise(self, sigma):
		'''Returns random noise with std sigma and with each pixel independent'''
		ffp_noise= {}
		for CCD in self.CCD_list:
			ffp_noise[CCD] = {}
			for ampno in range(1,17):
				ffp_noise[CCD][ampno] = np.random.normal(0,sigma,size=self.__getImagingShape__(CCD))
		return ffp_noise

	def getCCDCorrNoise(self, cov):
		'''Takes as input a 16x16 covariance matrix for a single CCD and returns noise for the 
		FP where CCDs are assumed to be independent of each other and normally distributed 
		within the CCD'''
		ffp_noise = {}
		for CCD in self.CCD_list:
			ffp_noise[CCD] = {}
			size = self.__getImagingShape__(CCD)
			noise = np.random.multivariate_normal(np.zeros(16), cov,size=size)
			for ampno in range(1,17):
				ffp_noise[CCD][ampno] = noise[:,:,ampno-1]
		return ffp_noise

	def getMultiCCDCorrNoise(self, covStack):
		'''Takes as input a 9x16x16 array of covariance matrices for the nine CCDs on a raft
		and returns noise for the FP where the CCDs are assumed to be independent of each other 
		and normally distributed within the CCD. Randomly shuffles order of covariance matrices
		on the raft'''
		ffp_noise = {}
		for raft in self.rafts:
			np.random.shuffle(covStack)
			for islot, slot in enumerate(self.slots):
				ccdid = '{}_{}'.format(raft,slot)
				ffp_noise[ccdid] = {}
				size = self.__getImagingShape__(ccdid)
				noise = np.random.multivariate_normal(np.zeros(16),covStack[islot,:,:], size=size)
				for ampno in range(1,17):
					ffp_noise[ccdid][ampno] = noise[:,:,ampno-1]
		return ffp_noise

