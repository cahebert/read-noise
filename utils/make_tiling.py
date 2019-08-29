import numpy as np
import lsst.geom as geom
import pickle

outFile = open('pointingList.obj', 'wb')

## Get  pointing angles

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

pickle.dump(pointingList, outFile)
