"""
2. Equalização de Histograma. (AQUI)
   Aplicar a equalização de histograma mostrando os histogramas inicial e final
   das imagens da aulas ch03-transformacao de intensidade
   (pollen, cameraman, etc).
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
image = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)

# Aplicar a equalização de histograma
equalized_image = cv2.equalizeHist(image)

# Mostrar a imagem original e a imagem equalizada
cv2.imshow('Original', image)
cv2.imshow('Equalizada', equalized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Plotar os histogramas
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(image.ravel(), 256, [0, 256])
plt.title('Histograma da Imagem Original')

plt.subplot(1, 2, 2)
plt.hist(equalized_image.ravel(), 256, [0, 256])
plt.title('Histograma da Imagem Equalizada')

plt.show()
