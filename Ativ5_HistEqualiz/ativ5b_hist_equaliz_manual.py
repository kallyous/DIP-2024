"""
2. Equalização de Histograma.
   Aplicar a equalização de histograma mostrando os histogramas inicial e final
   das imagens da aulas ch03-transformacao de intensidade
   (pollen, cameraman, etc).
Aqui os cáculos foram feitos sem usar a função pronta do cv2:
https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
"""

import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# Imagem default
path_img = 'ex1.jpg'

# Se foi fornecido outra imagem, pra usar com as dos slides da aula.
if len(sys.argv) > 1:
    path_img = sys.argv[1]

# Carrega imagem em escala de cinza
img_orig = cv.imread(path_img, cv.IMREAD_GRAYSCALE)

# TODO: Eu sei que isso me dá um histograma e as bins dele, falta estudar e
#  entender porque o tutorial fez dessa forma, quebrando a tupla.'''
hist, bins = np.histogram(img_orig.flatten(), 256, (0, 256))

# Soma cumulativa da distribuição de frequências (cumsum is not cum sum)
cdf = hist.cumsum()

# CDF normalizado
cdf_normalized = cdf * float(hist.max()) / cdf.max()

# plt.plot(cdf_normalized, color='b')
# plt.hist(img_orig.flatten(), 256, [0, 256], color='r')
# plt.xlim([0, 256])
# plt.legend(('cdf', 'histogram'), loc='upper left')
# plt.show()

cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')

img_eqlzd = cdf[img_orig]


""" SAÍDA """

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(img_orig.ravel(), 256, [0, 256])
plt.title('Histograma da Imagem Original')

plt.subplot(1, 2, 2)
plt.hist(img_eqlzd.ravel(), 256, [0, 256])
plt.title('Histograma da Imagem Equalizada')

plt.show()

cv.imshow('Original', img_orig)
cv.imshow('Equalized', img_eqlzd)

cv.waitKey(0)
cv.destroyAllWindows()
