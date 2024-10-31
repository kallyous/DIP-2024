import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



def implot(img: np.ndarray, figsize=(10, 5), title=""):
    """
    Plota ndarray como uma imagem em escala de cinza.

    Args:
        img:  Imagem a plotar.
        figsize (tupla widith, height):  Tamanho da figura.
        title (str):  Título para o plot.
    """
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray', aspect="equal")
    plt.title(title)
    plt.axis("off")
    plt.show()



def implotmany(images: list, n_cols=3, cmap="gray", endl=True):

    print("D.Type:", type(images[0][0][0]), "    Img.Shape:", images[0].shape, f"    {len(images)} imgs")

    # Pega qtd de imagens.
    n_img = len(images)

    # Sem imagens vai plotar o quê, porra!?
    if n_img < 1:
            raise ValueError("Lista de imagens a plotar está vazia.")
    
    # Calcula quantidade de linhas.
    n_lines = (n_img // n_cols)
    if n_img % n_cols > 0:
        n_lines += 1

    # Cria uma figura com subplots.
    fig, axes = plt.subplots(n_lines, n_cols, figsize=(14, 4 * n_lines))
    
    # Exibição em linhas e colunas.
    for i, img in enumerate(images):
        if n_lines == 1:
            ax = axes[i]
        else:
            ax = axes[i // n_cols, i % n_cols]  # Acessando o subplot correspondente
        ax.imshow(img, cmap=cmap)
        ax.axis("off")  # Remove os eixos, reduzindo espaço em branco desnecessário entre os sobplots.
    
    plt.tight_layout()  # Subplots mais próximos entre si.
    plt.show()

    if endl: print()



def imdetails(img: np.ndarray, title="Imagem", figsize=(16, 5)):

    # Calcula histograma e bins para set barplot com numpy.
    hist_np, bins_np = np.histogram(img.ravel(), bins=256, range=(0, 256))

    # Configura subplots de 1 linha e 2 colunas.
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    # Subplot 1: Imagem em Escala de Cinza
    axs[0].imshow(img, cmap='gray')  # Usar colormap 'gray'
    axs[0].set_title(title)
    axs[0].axis('off')  # Remover eixos para a imagem
    axs[0].set_aspect(aspect='equal')  # Ajusta o aspecto
    
    # Subplot 2: Histograma Numpy
    axs[1].bar(bins_np[:-1], hist_np, color='k', edgecolor='k')
    axs[1].set_title("Histograma")
    axs[1].set_xlim([0, 256])  # Limitar o eixo X para intensidades de 0 a 255
    axs[1].set_xlabel("Níveis de Cinza")
    axs[1].set_ylabel("Frequência de Pixels")
    
    # Ajustar o layout
    plt.tight_layout()  # Ajustar os espaçamentos
    plt.show()



def drawlines(image_set: list, lines_set: list, thickness=3, inplace=False):
    """Desenha as linhas fornecidas, em cópias das imagens fornecidas, e retorna essas cópias."""

    if not inplace:
        imset_draw = [img.copy() for img in image_set]
    else:
        imset_draw = image_set

    for i in range(len(lines_set)):
        for line, in lines_set[i]:
            rho = line[0]
            theta = line[1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(imset_draw[i], pt1, pt2, 255, thickness, cv.LINE_AA)
    return imset_draw
















