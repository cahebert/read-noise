import healpy as hp
import numpy as np
import healpix_util as hu
import sys

nside = 32 # nside of cell you want randoms for
nrand = 1000 # number of randoms you want to use
nest = True
cpix = int(sys.argv[1]) # or whatever healpix cell you want randoms for

dt = np.float
pmap = np.zeros(12*nside**2)
pmap[cpix] = 1
rdtype = np.dtype([('ra', dt), ('dec', dt)])

if nest:
    pmap       = hu.DensityMap('nest', pmap)
else:
    pmap       = hu.DensityMap('ring', pmap)
    
grand = np.zeros(nrand, dtype=rdtype)
grand['ra'], grand['dec'] = pmap.genrand(nrand, system='eq')

import fitsio
fitsio.write('/nfs/slac/g/ki/ki19/lsst/jrovee/randoms/hp/pixel%s.fits' % cpix, grand)
