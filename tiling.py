import numpy as np
import os
import glob
import healpy as hp
import lsst.afw.geom as geom
from lsst.obs.lsst import LsstCamMapper as camMapper
from lsst.obs.lsst.lsstCamMapper import getWcsFromDetector
from astropy.io import fits
        
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
inBoundsFast = np.vectorize(inBoundsFast)

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

### Get pointing angles
pointingList = []
currLat = 2

while(currLat < 89):
    # Go to center
    currLon = 90
    midPoint = geom.SpherePoint(currLon, currLat, geom.degrees)
    pointingList.append(midPoint)
    
    # Go right
    while(currLon < 180 - 2/np.cos(currLat*np.pi/180)):
        point = geom.SpherePoint(currLon, currLat, geom.degrees)
        if point.separation(pointingList[-1]).asDegrees() < 4:
            currLon += 0.2/np.cos(currLat*np.pi/180)
            continue
        pointingList.append(point)
        currLon += 4/np.cos(currLat*np.pi/180)
    
    # Go back to center
    currLon = 90 - 4/np.cos(currLat*np.pi/180)
    while(midPoint.separation(geom.SpherePoint(currLon, currLat, geom.degrees)).asDegrees() < 4):
        currLon -= 0.2/np.cos(currLat*np.pi/180)
    
    # Go left
    while(currLon > 2/np.cos(currLat*np.pi/180)):
        point = geom.SpherePoint(currLon, currLat, geom.degrees)
        if point.separation(pointingList[-1]).asDegrees() < 4:
            currLon -= 0.2/np.cos(currLat*np.pi/180)
            continue
        pointingList.append(point)
        currLon -= 4/np.cos(currLat*np.pi/180)
    
    # Move up and repeat
    currLat += 4
    
# Check for points that might overlap with the boundary and remove them (there are 2)
toRemove = []
for point in pointingList:
    for lat in np.arange(0,90,0.1):
        if point.separation(geom.SpherePoint(0,lat,geom.degrees)).asDegrees() < 2.2:
            toRemove.append(point)
            break
        if point.separation(geom.SpherePoint(180,lat,geom.degrees)).asDegrees() < 2.2:
            toRemove.append(point)
            break
for point in toRemove:
    pointingList.remove(point)

root = "/nfs/slac/des/fs1/g/sims/jderose/BCC/Chinchilla/Herd/Chinchilla-4/v1.9.2/addgalspostprocess/truth/truth"
fNamePattern = "Chinchilla-4_lensed.{}.fits"

### Write focal plane copies to fits files

camera = camMapper._makeCamera()

for iPoint, boresight in pointingList:
    pixels = getPixelsFromCenter(boresight.getRa().asDegrees(), boresight.getDec().asDegrees(), camera)
    wcsList = {detector : getWcsFromDetector(detector, boresight) for detector in camera}
    testFiles = [os.path.join(root,fNamePattern.format(pixel)) for pixel in pixels]
    hdulists = [fits.open(file) for file in testFiles]
    data = [hdulist[1].data for hdulist in hdulists]

    dummy = fits.open(os.path.join(root,fNamePattern.format(0)))

    endOfLast = 0
    for i, hdu in enumerate(data):
        ra = hdu['RA']
        dec = hdu['DEC']
        valid = hdu[inBoundsFast(ra, dec)]
        dummy[1].data[endOfLast:endOfLast+len(valid)] = valid
        endOfLast = len(valid)
    dummy[1].data = dummy[1].data[:endOfLast]

    dummy[1].data['TRA'] = np.array([boresight.getRa().asDegrees()])
    dummy[1].data['TDEC'] = np.array([boresight.getDec().asDegrees()])

    dummy.writeto('outputs/fpCopy{}.fits'.format(iPoint))