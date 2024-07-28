"""Carrega imagem em escala de cinza e plota seu histograma.
Compara as plotagens do pyplot e seaborn.
Lucas Carvalho Flores
"""


import numpy as np
import cv2


""" ENTRADA """

# Carregar a imagem em escala de cinza
image = cv2.imread("Osamu-Dazai-Bungo-Stray-Dogs.png",
                   cv2.IMREAD_REDUCED_COLOR_2)

# Verificar se a imagem foi carregada corretamente
if image is None:
    raise ValueError("Imagem não encontrada.")

# Exibir a imagem com OpenCV.
cv2.imshow('Original', image)


""" MULTIPLICAÇÃO POR ESCALAR """

'''
# Vetoriza função de multiplicação.
vectorized_multi = np.vectorize(lambda x: x * 1.25)

# Cria ndarray recipiente, para aplicar função em todos os canais
# da imagem original.
img_multi = np.zeros_like(image)

# Laço para iterar todos os canais da imagem.
for i in range(0, 3):
    img_multi[:, :, i] = vectorized_multi(image[:, :, i])
'''

# Faz a mesma coisa que o trecho de código anterior, e ainda por cima
# é segura contra estouro de valor.
img_multi = cv2.multiply(image, np.array([1.5, 1.5, 1.5]))

cv2.imshow('Multiplicacao', img_multi)


""" SOMA DE IMAGENS """

# Obter as dimensões da imagem
height, width = image.shape[:2]

# Verificar se a imagem é colorida ou cinza
if len(image.shape) == 3:
    # Imagem colorida (RGB)
    gray_image = np.full((height, width, 3), 64, dtype=np.uint8)
else:
    # Imagem grayscale
    gray_image = np.full((height, width), 128, dtype=np.uint8)

# Desenhar um círculo branco no centro da imagem
center_coordinates = (width // 2, height // 2)
radius = min(height, width) // 4
color = (128, 128, 128)  # Cinza claro
thickness = -1  # Preencher o círculo

cv2.circle(gray_image, center_coordinates, radius, color, thickness)

cv2.imshow('Bola em fundo cinza', gray_image)

'''Somar as duas imagens com cv2.sum() ao invés de usar image + gray_image
evita estouro de valores.'''
img_sum = cv2.add(image, gray_image)

cv2.imshow('Soma das imagens', img_sum)


""" FIM """

cv2.waitKey(0)
cv2.destroyAllWindows()
