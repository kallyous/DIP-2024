"""Carrega imagem em escala de cinza e plota seu histograma.
Compara as plotagens do pyplot e seaborn.
Lucas Carvalho Flores
"""


import cv2
import matplotlib.pyplot as plt
import seaborn as sns


""" ENTRADA """

# Carregar a imagem em escala de cinza
image = cv2.imread("Osamu-Dazai-Bungo-Stray-Dogs.png",
                   cv2.IMREAD_REDUCED_GRAYSCALE_2)

# Verificar se a imagem foi carregada corretamente
if image is None:
    raise ValueError("Imagem não encontrada.")

# Exibir a imagem
plt.imshow(image, cmap="gray")
plt.title("Imagem em Escala de Cinza")
plt.axis("off")
plt.show()


""" PYPLOT """

# Calcular o histograma. TODO: Estudar a função cv3.calcHist()
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# Plotar o histograma
plt.figure()
plt.title("Histograma com Matplotlib")
plt.xlabel("Intensidade de Pixel")
plt.ylabel("Frequência")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()


""" SEABORN """

# Seaborn funciona melhor com dados formatados, por isso transformamos
# a imagem em um array unidimensional.
image_flatten = image.flatten()

# Plotar o histograma
plt.figure()
plt.title("Histograma com Seaborn")
plt.xlabel("Intensidade de Pixel")
plt.ylabel("Frequência")
sns.histplot(image_flatten, bins=256, kde=False, color="blue")
plt.xlim([0, 256])
plt.show()
