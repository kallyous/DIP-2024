"""Carrega imagem em escala de cinza e plota seu histograma.
Compara as plotagens do pyplot e seaborn.
Lucas Carvalho Flores
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


""" ENTRADA """

# Carregar a imagem em escala de cinza
image = cv2.imread('Osamu-Dazai-Bungo-Stray-Dogs.png',
                   cv2.IMREAD_REDUCED_GRAYSCALE_2)

# Verificar se a imagem foi carregada corretamente
if image is None:
    raise ValueError("Imagem não encontrada.")

# Exibir a imagem
plt.imshow(image, cmap='gray')
plt.title('Imagem em Escala de Cinza')
plt.axis('off')
plt.show()


""" MATPLOTLIB """

# Transformar a imagem em um array unidimensional
image_flatten = image.flatten()

# Plotar o histograma
plt.figure()
plt.title('Distribuição de Intensidade de Pixel com Matplotlib')
plt.xlabel('Intensidade de Pixel')
plt.ylabel('Frequência')
plt.hist(image_flatten, bins=256, range=[0, 256], color='gray', density=True)

# Adicionar uma linha representando a distribuição normal
mean = image_flatten.mean()
std = image_flatten.std()
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
plt.plot(x, p, 'k', linewidth=2)
plt.savefig('ativ1c_plot_matplotlib.jpg')
plt.show()


""" SEABORN """

# Transformar a imagem em um array unidimensional
image_flatten = image.flatten()

# Plotar o histograma com seaborn
plt.figure()
plt.title('Distribuição de Intensidade de Pixel com Seaborn')
plt.xlabel('Intensidade de Pixel')
plt.ylabel('Frequência')
sns.histplot(image_flatten, bins=256, kde=True, color='gray')

# Adicionar uma linha representando a distribuição normal
mean = image_flatten.mean()
std = image_flatten.std()
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
plt.plot(x, p, 'k', linewidth=2)
plt.savefig('ativ1c_plot_seaborn.jpg')
plt.show()
