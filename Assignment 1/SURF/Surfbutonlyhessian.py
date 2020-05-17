
import numpy as np
import cv2
from PIL import Image

# Load the image
# # Detect the SURF key points
# surf = cv2.xfeatures2d.SURF_create(hessianThreshold=50000, upright=True, extended=True)
# keyPoints, descriptors = surf.detectAndCompute(gray, None)

def Hessian(img,k,sigma):
	imagevars = []
	n = np.ceil(sigma*6)
	y,x = np.ogrid[-n//2:n//2+1,-n//2:n//2+1]
	Convfilter1 = ( -1 + (x**2)/(sigma**2) ) * ( np.exp(-( ((x**2) + (y**2)) / ( sigma**2 )))) / 2*(np.pi)*( sigma**4 ) 
	Convfilter2 = ( -1 + (y**2)/(sigma**2) ) * ( np.exp(-( ((x**2) + (y**2)) / ( sigma**2 )))) / 2*(np.pi)*( sigma**4 ) 
	Convfilter3 = ( (x*y)/2*(np.pi)*(sigma**6) ) * ( np.exp(-( ((x**2) + (y**2)) / ( sigma**2 )))) 
	image1 = np.square(cv2.filter2D(img, ddepth = -1, kernel = Convfilter1)) 
	image2 = np.square(cv2.filter2D(img, ddepth = -1, kernel = Convfilter2))
	image3 = np.square(cv2.filter2D(img, ddepth = -1, kernel = Convfilter3))
	imagevars.append(image1)
	imagevars.append(image2)
	imagevars.append(image3)
	log = np.array([i for i in imagevars])
	return log

def blobdetection(log, k, sigma):
    coordinates = [] 
    for i in range(1,64):
        for j in range(1,64):
            logsizes = log[:,i-1:i+2,j-1:j+2]
            peakvalue = np.max(logsizes)
            #print(peakvalue)
            if peakvalue >= 100:
                a = np.unravel_index(logsizes.argmax(), logsizes.shape)
                print(a)
                # ycoord = y+i-1
                # xcoord = x+j-1
                # radius = (k**r)*sigma*1.414
                # coordinates.append((xcoord,ycoord,radius)) 
    coordinateslist = list(set(coordinates))
    return coordinateslist

img = cv2.imread('test.jpg')
img = cv2.resize(img, (64,64))
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Hes = Hessian(img, 1.4, 1)
# print(Hes)
coordinates = blobdetection(Hes, k = 1.4, sigma = 1)
print(coordinates)
