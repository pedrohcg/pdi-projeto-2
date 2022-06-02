import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import skimage.util
import os.path


def dct1d(img):
    img = img.flatten(order='C')
    n = img.size
    i = k = sum = 0

    dct = np.zeros(n)

    for k in range(k, n-1, 1):
        for i in range(i, n-1, 1):
            sum += img[i]*math.cos((2*math.pi*k*(i/(2*n))) + (k*math.pi/(2*n)))
        i = 0

        if(k == 0):
            ck = np.power(1/2, 1/2)
        else:
            ck = 1

        dct[k] = np.power(2/n, 1/2)*ck*sum
        
        sum = 0

    return dct

def idct1d(img):
    img = img.flatten(order='C')

    n = img.size
    i = k = sum = 0

    idct = np.zeros(n)

    for k in range(k, n-1, 1):
        for i in range(i, n-1, 1):
            if(i == 0):
                ck = np.power(1/2, 1/2)
            else:
                ck = 1
            sum += ck*img[i]*math.cos((2*math.pi*i*(k/(2*n))) + (i*math.pi/(2*n)))
        i = 0

        idct[k] = np.power(2/n, 1/2)*sum

        sum = 0
    
    return idct

def dct2d(img):
    h = w = 8
    i = j = k = l = sum = 0
    ci = cj = 0
    m = n = 8

    dct = np.zeros(img.shape)

    for i in range(i, h, 1):
        if(i == 0):
            ci = 1/math.sqrt(2)
        else:
            ci = 1

        for j in range(j, w, 1):

            if(j == 0):
                cj = 1/math.sqrt(2)
            else:
                cj = 1

            sum = 0
            for k in range(k, m, 1):
                for l in range(l, n, 1):
                    sum += (img[k, l]*np.cos(((2*k+1)*i*math.pi)/(2*m))*np.cos(((2*l+1)*j*math.pi)/(2*n)))
                l = 0
            k = 0
            dct[i, j] = round((1/np.sqrt((2*n)))*ci*cj*sum)
            
        j = 0
    
    return dct

def idct2d(img):
    h = w = 8
    i = j = k = l = sum = 0
    ci = cl = 0
    m = n = 8

    dct = np.empty_like(img)

    for i in range(i, h, 1):
        for j in range(j, w, 1):
            sum = 0
            for k in range(k, m, 1):
                if(k == 0):
                    ck = 1/math.sqrt(2)
                else:
                    ck = 1
                for l in range(l, n, 1):
                    if(l == 0):
                        cl = 1/math.sqrt(2)
                    else:
                        cl = 1

                    sum += ck*cl*(img[k, l]*np.cos(((2*i+1)*k*math.pi)/(2*m))*np.cos(((2*j+1)*l*math.pi)/(2*n)))
                l = 0
            k = 0
            dct[i, j] = round((1/np.sqrt((2*n)))*sum)
            
        j = 0
    
    return dct