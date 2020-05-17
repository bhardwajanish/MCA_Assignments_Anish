from PIL import Image
import numpy as np
import json
import os

def corellindex(color):
    return 9*color[0] + 3*color[1] + 1*color[2] 

def normalizeRGB(pixel):
    pixel1 = pixel[0]/256
    pixel2 = pixel[1]/256
    pixel3 = pixel[2]/256
    R = normalize(pixel1)
    G = normalize(pixel2)
    B = normalize(pixel3)
    return(R,G,B)

def normalize(val):
    if(val<=0.33):
        return 0
    if(val<=0.66):
        return 1
    if(val<=1):
        return 2

def corell(img):
    corellogram = np.zeros([27,2])
    dcheck = [1,2]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c = (i,j,k)
                for d in dcheck:
                    index = corellindex(c)
                    value = calcco(d, c, img)
                    if d==1:
                        i2 = 0
                    else:
                        i2 = 1
                    corellogram[index][i2] = int(value)
    return corellogram

def calcco(d, c, i):
    final = 0
    length = len(i)
    for x in range(0,length):
        for y in range(0,length):
            pix = i[x][y]
            if(pix[0] == c[0] and pix[1] == c[1] and pix[2] == c[2]):
                final+=checkpix(x,y+d,c,i)
                final+=checkpix(x,y-d,c,i)
                final+=checkpix(x-d,y,c,i)
                final+=checkpix(x+d,y,c,i)
    return final

def checkpix(x,y,c,i):
    if((x<0 or y<0) or (x>=len(img) or y>=len(img[0]))):
        return 0
    pix = i[x][y]
    if(pix[0] == c[0] and pix[1]==c[1] and pix[2]==c[2]):
        return 1
    else:
        return 0

co = {}

for f in os.listdir('images'):
    img = Image.open(f)
    img = img.resize((32,32))
    imagename = (str)(f)
    print(imagename)
    img = np.array(img)
    cimg = img
    for i in cimg:
        for j in i:
            RGB  = normalizeRGB(j)
            j[0] = RGB[0]
            j[1] = RGB[1]
            j[2] = RGB[2]
    corellogram = corell(img)
    co[imagename] = corellogram.tolist()
with open('data3.json', 'w') as outfile:
    json.dump(co, outfile)
