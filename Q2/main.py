import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import skimage.util
import os.path

script_dir = os.path.dirname(os.path.abspath(__file__))
im = Image.open(os.path.join(script_dir, '../imagens/lena128.png')).convert('L')

a = np.array(im)

def dct1d(img):
    img = img.flatten(order='C')
    n = img.size
    i = k = sum = 0

    dct = np.zeros(n)

    for k in range(k, n, 1):
        for i in range(i, n, 1):
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

    for k in range(k, n, 1):
        for i in range(i, n, 1):
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
    h, w = img.shape
    i = j = k = l = sum = 0
    ci = cj = 0
    m = h
    n = w

    dct = np.zeros(img.shape)

    for i in range(i, h, 1):
        if(i == 0):
            ci = np.power(1/2, 1/2)
        else:
            ci = 1

        for j in range(j, w, 1):

            if(j == 0):
                cj = np.power(1/2, 1/2)
            else:
                cj = 1

            sum = 0
            for k in range(k, m, 1):
                for l in range(l, n, 1):
                    sum += (img[k, l]*np.cos(((2*k+1)*i*math.pi)/(2*m))*np.cos(((2*l+1)*j*math.pi)/(2*n)))
                l = 0
            k = 0
            dct[i, j] = round((ci*cj*sum)/np.sqrt((2*n)))
        j = 0
    i = 0

    return dct

def idct2d(img):
    h, w = img.shape
    i = j = k = l = sum = 0
    ci = cl = 0
    m = h
    n = w

    dct = np.zeros((h, w))

    for i in range(i, h, 1):
        for j in range(j, w, 1):
            sum = 0
            for k in range(k, m, 1):
                if(k == 0):
                    ck = np.power(1/2, 1/2)
                else:
                    ck = 1
                for l in range(l, n, 1):
                    if(l == 0):
                        cl = np.power(1/2, 1/2)
                    else:
                        cl = 1
                   
                    sum += ck*cl*(img[k, l]*np.cos(((2*i+1)*k*math.pi)/(2*m))*np.cos(((2*j+1)*l*math.pi)/(2*n)))
                l = 0
            k = 0
            dct[i, j] = round(sum/(np.sqrt((2*n))))
        j = 0
    i = 0
    
    return dct

def butterworth(img, fc):
    h, w = img.shape
    i = j = 0
    n = 3

    for i in range(i, h, 1):
        for j in range(j, w, 1):
            euclidian_distance = np.sqrt(np.power(i, 2) + np.power(j, 2))

            if((i+1)*(j+1) < fc):
                img[i, j] = img[i, j]*(1/(np.sqrt(1+np.power(euclidian_distance/fc, 2*n))))
            else:
                img[i, j] = 0
        j = 0
    i = 0
    
    return

def exec(a):
    h, w = a.shape

    print(a)

    resultado_dct = dct1d(a)

    resultado_dct = np.reshape(resultado_dct, (h, w))
    
    butterworth(resultado_dct, 600)

    resultado_idct = idct1d(resultado_dct)

    resultado_idct = np.reshape(resultado_idct, (h, w))

    print(resultado_idct)

    img_resultante_idct = Image.fromarray(resultado_idct.astype(np.uint8))

    img_resultante_idct.save("../resultados/q2/butterworth-1d.png")

    return

def exec_2(a):
    h, w = a.shape
    offset = h/8
    resultado_dct = np.zeros(a.shape)
    
    resultado_dct = dct2d(a)    

    print(a[0,0])
    print(resultado_dct)
    
    butterworth(resultado_dct, 120)
    
    resultado_idct = np.zeros(a.shape)

    resultado_idct = idct2d(resultado_dct)

    resultado_idct = resultado_idct/offset
     
    img_resultante_idct2d = Image.fromarray(resultado_idct.astype(np.uint8))

    img_resultante_idct2d.save("../resultados/q2/butterworth-2d.png")

    return

exec(a)
#exec_2(a)