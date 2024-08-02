"""
2. Equalização de Histograma.
   Aplicar a equalização de histograma mostrando os histogramas inicial e final
   das imagens da aulas ch03-transformacao de intensidade
   (pollen, cameraman, etc).
Aqui usamos a função pronta do cv2 pra comparar os resutlados.
https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
"""

import sys
import cv2
import matplotlib.pyplot as plt


# Imagem default
path_img = 'ex1.jpg'

# Se foi fornecido outra imagem, pra usar com as dos slides da aula.
if len(sys.argv) > 1:
    path_img = sys.argv[1]

# Carregar a imagem em escala de cinza
img_orig = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)

# Aplicar a equalização de histograma
img_eqlzd = cv2.equalizeHist(img_orig)

# Mostrar a imagem original e a imagem equalizada
cv2.imshow('Original', img_orig)
cv2.imshow('Equalizada', img_eqlzd)

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
