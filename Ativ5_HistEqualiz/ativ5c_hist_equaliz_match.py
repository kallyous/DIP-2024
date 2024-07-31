"""
3. Aplicar a correspondência de histograma (matching) na imagem
   do slide 45 (moon).

4. Aplicar a equalização local de histograma conforme apresentado
   no slide 50 da mesma aula.
"""


import cv2

# Imagem default
path_img = 'fig-3-23.png'

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