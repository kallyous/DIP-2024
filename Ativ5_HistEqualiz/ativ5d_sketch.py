"""
4. Aplicar a equalização local de histograma conforme apresentado
   no slide 50 da mesma aula.
"""

import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# Imagem default
path_img = 'Fig0326(a)(embedded_square_noisy_512).tif'

# Se foi fornecido outra imagem, para usar com as dos slides da aula.
if len(sys.argv) > 1:
    path_img = sys.argv[1]

# Carrega imagem em escala de cinza
img_orig = cv.imread(path_img, cv.IMREAD_GRAYSCALE)

# Vai receber a imagem equalizada
img_eqlzd = img_orig.copy()

# Passa por todos os píxeis não-borda e seta preto na imagem de destino.
height, width = img_orig.shape
for y in range(1, height-1):
    for x in range(1, width-1):
        kernel = img_orig[y-1:y+2, x-1:x+2]
        kernel = cv.equalizeHist(kernel)
        img_eqlzd[y-1:y+2, x-1:x+2] = kernel[1, 1]


""" SAÍDA """

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(img_orig.ravel(), 256, (0, 256))
plt.title('Histograma da Imagem Original')

plt.subplot(1, 2, 2)
plt.hist(img_eqlzd.ravel(), 256, (0, 256))
plt.title('Histograma da Imagem Equalizada')

plt.show()

cv.imshow('Original', img_orig)
cv.imshow('Equalized', img_eqlzd)

cv.waitKey(0)
cv.destroyAllWindows()
