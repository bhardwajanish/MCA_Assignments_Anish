import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def LoG(img, k, sigma):
    imagevars = []
    for i in range(0,3):
        filtersize = (k**i)*sigma
        n = round(filtersize*4)
        y,x = np.ogrid[-n//2:n//2+1,-n//2:n//2+1]
        filterlog = (-1/(3.14*sigma**4))*(1 - ((x**2 + y**2)/(2*sigma**2)) )*(np.exp(-(((x**2)+(y**2))/(2.*(sigma**2)))))
        image = np.square(cv2.filter2D(img, ddepth = -1, kernel = filterlog)) 
        imagevars.append(image)
    log = np.array([i for i in imagevars])
    return log


def blobdetection(log, k, sigma):
    coordinates = [] 
    for i in range(1,64):
        for j in range(1,64):
            logsizes = log[:,i-1:i+2,j-1:j+2]
            logsizes1, logsizes2, logsizes3 = logsizes
            x = [logsizes1[1][1], logsizes2[1][1], logsizes3[1][1]]
            peakvalue = np.max(x)
            if peakvalue >= 0.05:
                r,y,x = np.unravel_index(logsizes.argmax(), (3,3,3))
                ycoord = y+i-1
                xcoord = x+j-1
                radius = (k**r)*sigma*1.414
                coordinates.append((xcoord,ycoord,radius)) 
    coordinateslist = list(set(coordinates))
    return coordinateslist

keypoints = {}
for f in os.listdir('images'):
    f = str(f)
    print(f)
    img = cv2.imread(f, 0)
    img = cv2.resize(img, (64,64))
    img2 = img/255
    log = LoG(img2, k = 1.4, sigma = 1.12)
    coordinates = blobdetection(log, k = 1.4, sigma = 1.12)
    # _, a = plt.subplots()
    # a.imshow(img)
    imagename = (str)(f)
    for blob in coordinates:
        x,y,r = blob
        # c = plt.Circle((x, y), r, color='red', linewidth=0.5, fill = False)
        # a.add_patch(c)
        x = (int)(x)
        y = (int)(y)
        if imagename in keypoints:
            keypoints[imagename].append((x,y))
        else:
            keypoints[imagename] = [(x,y)]
# a.plot()  
# plt.show()
with open('data2.json', 'w') as outfile:
    json.dump(keypoints, outfile)
