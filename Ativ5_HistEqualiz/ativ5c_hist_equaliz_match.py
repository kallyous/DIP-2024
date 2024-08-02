"""
3. Aplicar a correspondência de histograma (matching) na imagem
   do slide 45 (moon).
Aconteceu que a equalização normal não resultou na imagem de baixo contraste
apresentado nos slides, essa imagem não precisa de correspondência de
histograma.
TODO: Atualizar com imagem que precise de histogram matching.
"""

import matplotlib.pyplot as plt
import cv2

# Imagem default
path_img = 'Fig0323(a)(mars_moon_phobos).tif'

# Carregar a imagem em escala de cinza
img_orig = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original', img_orig)

# Equaliza com a função do cv2 mesmo
img_eqlzd = cv2.equalizeHist(img_orig)
cv2.imshow('Equalized', img_eqlzd)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Plotar os histogramas
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(img_orig.ravel(), 256, [0, 256])
plt.title('Histograma da Imagem Original')

plt.subplot(1, 2, 2)
plt.hist(img_eqlzd.ravel(), 256, [0, 256])
plt.title('Histograma da Imagem Equalizada')

plt.show()
