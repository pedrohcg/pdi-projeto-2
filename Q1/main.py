import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import os.path

script_dir = os.path.dirname(os.path.abspath(__file__))
im = Image.open(os.path.join(script_dir, '../imagens/lena256.png')).convert('L')

a = np.array(im)
#a = np.ones((8, 8))

#a = a*255

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

def idct(img):
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

def histograma(img):
    h, w = img.shape
    i = j = 0
    maior = np.amax(img)
    menor = np.amin(img)

    #print(maiorR, maiorG, maiorB)
    resultado = np.zeros([h, w], dtype='uint8') 

    for i in range(i, h, 1):
        for j in range(j, w, 1):
            tr = np.round_(np.multiply(np.divide(img[i, j] - menor, maior - menor), 254))
            resultado[i, j] = tr
        j = 0

    return resultado


h, w = a.shape

resultado_dct = dct1d(a)

print("Resultado DC: ",resultado_dct[0])
#Resultado DC:  31883.191406250007

resultado_dct[0] = 0

plt.plot(resultado_dct)
x = np.arange(0, resultado_dct.size)
plt.xlabel("Pixel")
plt.ylabel("Valor")
plt.plot(x, resultado_dct, color="blue")
plt.show()

resultado_dct = np.reshape(resultado_dct, (h, w))

img_resultante_dct = Image.fromarray(resultado_dct.astype(np.uint8))

img_resultante_dct.save("../resultados/q1/dct.png")

resultado_idct = idct(resultado_dct)

resultado_idct = np.reshape(resultado_idct, (h, w))

resultado_idct = histograma(resultado_idct)

#print(resultado_idct)

img_resultante_idct = Image.fromarray(resultado_idct.astype(np.uint8))

img_resultante_idct.save("../resultados/q1/idct.png")



