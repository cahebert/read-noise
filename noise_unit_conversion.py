import numpy as np

area                   = 6.423
collectionTime         = 30.
h                      = 6.62607004e-34
c                      = 1.e8
wavelength, throughput = np.genfromtxt('rbandThroughput.csv', unpack=True); wavelength *= 1.e-9
freq                   = c / wavelength
dfreq                  = -1 * np.diff(freq, prepend=0)
Jy_to_SI               = 1.e-26
nMgy_to_Jy             = 3.631e-6

nMgy_to_ADU = nMgy_to_Jy * Jy_to_SI * area * collectionTime * np.dot(throughput,dfreq/freq) / h
ADU_to_nMgy = 1 / nMgy_to_ADU

print(1/ADU_to_nMgy)
