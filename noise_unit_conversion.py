import numpy as np

area                   = 6.423
collectionTime         = 30.
h                      = 6.62607004e-34
c                      = 1.e8
wavelength, throughput = np.genfromtxt('rbandThroughput.csv', unpack=True); wavelength *= 1.e-9
freq                   = c / wavelength
dfreq                  = -1 * np.diff(freq, prepend=0)

factor = 3.631 * 1.e-6 * 1.e-26 * area * collectionTime * np.dot(throughput,dfreq/freq) / h

print(1/factor)
