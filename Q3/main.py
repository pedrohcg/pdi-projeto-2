import numpy as np
from scipy.io import wavfile
import math
from PIL import Image
import matplotlib.pyplot as plt
import skimage.util
import os.path

f = wavfile.read('../imagens/MaisUmaSemana.wav')

audio = f[1]

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

def butterworth(frequency, fc):
    h = frequency.size
    i = 0
    n = 3

    fa = 1/(h-1)
    f1 = fa/(2*(h-1))
    fchz = f1*fc
 
    for i in range(i, h, 1):
        if(i < fc):
            frequency[i] = frequency[i]*(1/np.sqrt((1 + (np.power(((f1*(i+1))/fchz), 2*n)))))
        else:
            frequency[i] = 0
    
    return

def plot(y):    
    x = np.arange(0, y.size)
    plt.xlabel("Pixel")
    plt.ylabel("Valor")
    plt.plot(x, y, color="blue")
    plt.show()

#plot(audio)

resultado_dct = dct1d(audio)
butterworth(resultado_dct, 16000)
plot(resultado_dct)
resultado_idct = idct1d(resultado_dct)
#print(resultado_idct)
wavfile.write('../resultados/q3/MaisUmaSemana16000.wav', f[0], resultado_idct.astype(np.int16))
#print(resultado_idct.astype(np.int16))
plot(resultado_idct.astype(np.int16))
"""a = wavfile.read('../resultados/q3/MaisUmaSemana480-2.wav')
print(a[1])
plot(f)"""