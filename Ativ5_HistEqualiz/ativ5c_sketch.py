"""
3. Aplicar a correspondência de histograma (matching) na imagem
   do slide 45 (moon).
"""

import matplotlib.pyplot as plt
import cv2

# Imagem default
path_img = 'Fig0323(a)(mars_moon_phobos).tif'

# x1 = 290
# y1 = 10
# x2 = 600
# y2 = 460

# Carregar a imagem em escala de cinza
img_orig = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original', img_orig)

img_eqlzd = cv2.equalizeHist(img_orig)
cv2.imshow('Equalized', img_eqlzd)

# Recorta pra foto contida na imagem.
# image = image[y1:y2, x1:x2]

cv2.waitKey(0)
cv2.destroyAllWindows()

# Plotar os histogramas
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(img_orig.ravel(), 256, (0, 256))
plt.title('Histograma da Imagem Original')

plt.subplot(1, 2, 2)
plt.hist(img_eqlzd.ravel(), 256, (0, 256))
plt.title('Histograma da Imagem Equalizada')

plt.show()
