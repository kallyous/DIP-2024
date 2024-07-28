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
                   cv2.IMREAD_REDUCED_COLOR_2)

# Verificar se a imagem foi carregada corretamente
if image is None:
    raise ValueError("Imagem não encontrada.")

# Exibir a imagem com OpenCV.
# cv2.imshow("Original", image)
# cv2.waitKey(0)


""" PYPLOT """

# Converter a imagem de BGR para RGB para exibição correta com matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Exibir a imagem com matplotlib.
plt.imshow(image_rgb)
plt.title("Imagem Colorida")
plt.axis("off")
plt.show()

# Cores para cada canal
colors = ("b", "g", "r")
channel_ids = (0, 1, 2)

# Plotar o histograma para cada canal de cor
plt.figure()
plt.title("Histograma com Matplotlib")
plt.xlabel("Intensidade de Pixel")
plt.ylabel("Frequência")

'''Usando a imagem original, não a modificada para exibir com matplotlib,
calcula histograma.'''
for channel_id, color in zip(channel_ids, colors):
    hist = cv2.calcHist([image], [channel_id], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.show()


""" SEABORN """

# Transformar cada canal em um array unidimensional e plotar o histograma
colors = ("blue", "green", "red")
channel_ids = (0, 1, 2)
channel_names = ("Blue", "Green", "Red")

plt.figure()
plt.title("Histograma com Seaborn")
plt.xlabel("Intensidade de Pixel")
plt.ylabel("Frequência")

for channel_id, color, channel_name in zip(channel_ids, colors, channel_names):
    # Extrair o canal específico
    channel = image[:, :, channel_id]
    channel_flatten = channel.flatten()

    # Plotar o histograma
    sns.histplot(channel_flatten, bins=256, kde=False,
                 color=color, label=channel_name)

plt.legend()
plt.xlim([0, 256])
plt.show()


""" FIM """

cv2.destroyAllWindows()
