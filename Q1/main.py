import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import skimage.util
import os.path

script_dir = os.path.dirname(os.path.abspath(__file__))
im = Image.open(os.path.join(script_dir, '../imagens/lena128.png')).convert('L')

a = np.array(im)

#a = np.ones(400)
#a = a*255

"""a = [[140, 144, 147, 140, 140, 155, 179, 175],
[144, 152, 140, 147, 140, 148, 167, 179],
[152, 155, 136, 167, 163, 162, 152, 172],
[168, 145, 156, 160, 152, 155, 136, 160],
[162, 148, 156, 148, 140, 136, 147, 162],
[147, 167, 140, 155, 155, 140, 136, 162],
[136, 156, 123, 167, 162, 144, 140, 147],
[148, 155, 136, 155, 152, 147, 147, 136]]"""

#a = np.reshape(a, (8, 8))
#print(a)

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


def aproximacao(img, n):
    img_array = img.flatten(order="C")

    img_ordenada = np.sort(img_array)

    maisImportante = img_ordenada[n]
    resultado = np.copy(img_array)
    i = 0

    for i in range(i, img_array.size, 1):
        if(img_array[i] < maisImportante):
            resultado[i] = 0

    return resultado

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

def exec_dct1d(a):
    h, w = a.shape

    resultado_dct = dct1d(a)

    print("Resultado DC: ",resultado_dct[0])
    #Resultado DC 256:  31883.191406250007
    #Resultado DC 128:  15912.953125000004

    resultado_aprox = aproximacao(resultado_dct, 1500)

    resultado_dct = np.reshape(resultado_dct, (h, w))

    resultado_dct_hist = histograma(resultado_dct)

    resultado_dct_hist = resultado_dct_hist.flatten(order="C")

    resultado_dct_hist[0] = 0
    
    plt.plot(resultado_dct_hist)
    x = np.arange(0, resultado_dct_hist.size)
    plt.xlabel("Pixel")
    plt.ylabel("Valor")
    plt.plot(x, resultado_dct_hist, color="blue")
    plt.show()

    img_resultante_dct = Image.fromarray(resultado_dct.astype(np.uint16))

    img_resultante_dct.save("../resultados/q1/dct.png")

    resultado_idct = idct1d(resultado_dct)
    resultado_idct_aprox = idct1d(resultado_aprox)

    resultado_idct = np.reshape(resultado_idct, (h, w))
    resultado_idct_aprox = np.reshape(resultado_idct_aprox, (h, w))

    resultado_idct = histograma(resultado_idct)
    resultado_idct_aprox = histograma(resultado_idct_aprox)

    #print(resultado_idct)

    img_resultante_idct = Image.fromarray(resultado_idct.astype(np.uint8))
    img_resultante_aprox = Image.fromarray(resultado_idct_aprox.astype(np.uint8))

    img_resultante_idct.save("../resultados/q1/idct.png")
    img_resultante_aprox.save("../resultados/q1/imagem_aprox.png")

    return

def exec_dct2d(a):
    h, w = a.shape
    img_blocks = skimage.util.view_as_blocks(a ,block_shape=(8, 8))
    b, c, d, e = img_blocks.shape
    i = j = k = l = 0
    img_blocks_dct = np.zeros(img_blocks.shape)
    
    for i in range (i, b, 1):
        for j in range (j, c, 1):
            img_blocks_dct[i, j] = dct2d(img_blocks[i, j])
        j = 0
    i = 0 
    
    #print("dct", img_blocks_dct[0, 0])

    resultado_dct = np.zeros(a.shape)

    for i in range(i, b, 1):
        for j in range(j, c, 1):
            for k in range(k, d, 1):
                for l in range(l, e, 1):
                    resultado_dct[(i*8+k), (j*8)+l] = img_blocks_dct[i, j][k, l]
                l = 0
            k = 0
        j = 0
    i = 0

    print("Resultado DC: ",resultado_dct[0, 0])
    #Resultado DC:  1270.0

    resultado_dct_hist = histograma(resultado_dct)

    resultado_dct_hist[0] = 0

    resultado_dct_hist = resultado_dct_hist.flatten(order="C")
    
    plt.plot(resultado_dct_hist)
    x = np.arange(0, resultado_dct_hist.size)
    plt.xlabel("Pixel")
    plt.ylabel("Valor")
    plt.plot(x, resultado_dct_hist, color="blue")
    plt.show()

    resultado_aprox = aproximacao(resultado_dct, 1500)
    resultado_aprox = np.reshape(resultado_aprox, (h, w))
    img_blocks_aprox = skimage.util.view_as_blocks(resultado_aprox, block_shape=(8, 8))

    img_resultante_dct2d = Image.fromarray(resultado_dct.astype(np.uint16))

    img_resultante_dct2d.save("../resultados/q1/dct2d.png")

    img_blocks_idct = np.zeros(img_blocks.shape)
    img_blocks_idct_aprox = np.zeros(img_blocks_aprox.shape)

    for i in range (i, b, 1):
        for j in range (j, c, 1):
            img_blocks_idct[i, j] = idct2d(img_blocks_dct[i, j])
            img_blocks_idct_aprox[i, j] = idct2d(img_blocks_aprox[i, j])
        j = 0
    i = 0 
    
    #print("idct", img_blocks_idct[0, 0])

    resultado_idct = np.zeros(a.shape)
    resultado_idct_aprox = np.zeros(a.shape)

    for i in range(i, b, 1):
        for j in range(j, c, 1):
            for k in range(k, d, 1):
                for l in range(l, e, 1):
                    resultado_idct[(i*8+k), (j*8)+l] = img_blocks_idct[i, j][k, l]
                    resultado_idct_aprox[(i*8+k), (j*8)+l] = img_blocks_idct_aprox[i, j][k, l]
                l = 0
            k = 0
        j = 0
    i = 0

    #print(resultado_idct[0])

    img_resultante_idct2d = Image.fromarray(resultado_idct.astype(np.uint8))
    img_resultante_idct2d_aprox = Image.fromarray(resultado_idct_aprox.astype(np.uint8))

    img_resultante_idct2d.save("../resultados/q1/idct2d.png")
    img_resultante_idct2d_aprox.save("../resultados/q1/idct2d_aprox.png")

    return

exec_dct1d(a)
#exec_dct2d(a)



