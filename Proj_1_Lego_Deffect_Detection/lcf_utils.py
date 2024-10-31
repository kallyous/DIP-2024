
import numpy as np
from matplotlib import pyplot as plt



def plot_img_gray(img: np.ndarray, figsize=(10, 5), title='Imagem Grayscale'):
    """
    Plota ndarray como uma imagem em escala de cinza.

    Args:
        img (np.ndarray):  Imagem a plotar.
        figsize (tupla widith, height):  Tamanho da figura.
        title (str):  Título para o plot.
    """
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()



# implot
def plot_img_gray_hist(img: np.ndarray, title="Imagem", figsize=(16, 5)):

    # Calcula histograma e bins para set barplot com numpy.
    hist_np, bins_np = np.histogram(img.ravel(), bins=256, range=(0, 256))

    # Configura subplots de 1 linha e 2 colunas.
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    # Subplot 1: Imagem em Escala de Cinza
    axs[0].imshow(img, cmap='gray')  # Usar colormap 'gray'
    axs[0].set_title(title)
    axs[0].axis('off')  # Remover eixos para a imagem
    axs[0].set_aspect(aspect='auto')  # Ajusta o aspecto
    
    # Subplot 2: Histograma Numpy
    axs[1].bar(bins_np[:-1], hist_np, color='k', edgecolor='k')
    axs[1].set_title("Histograma")
    axs[1].set_xlim([0, 256])  # Limitar o eixo X para intensidades de 0 a 255
    axs[1].set_xlabel("Níveis de Cinza")
    axs[1].set_ylabel("Frequência de Pixels")
    
    # Ajustar o layout
    plt.tight_layout()  # Ajustar os espaçamentos
    plt.show()



# imcompare
def compare_img_gray(img_a: np.ndarray,
                     img_b: np.ndarray,
                     title_a="Imagem A",
                     title_b="Imagem B",
                     figsize=(15,8)):

    bar_color = '#555555'
    
    # Calcula histograma e bins para set barplot com numpy.
    hist_a, bins_a = np.histogram(img_a.ravel(), bins=256, range=(0, 256))
    hist_b, bins_b = np.histogram(img_b.ravel(), bins=256, range=(0, 256))

    # Configura subplot para duas linhas e duas colunas.
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # Subplot 1: Imagem A em Escala de Cinza
    axs[0, 0].imshow(img_a, cmap='gray')  # Usar colormap 'gray'
    axs[0, 0].set_title(title_a)          # Título de A
    axs[0, 0].axis('off')                 # Remove eixos para imagem
    axs[0, 0].set_aspect(aspect='auto')   # Ajusta o aspecto
    
    # Subplot 2: Histograma de A
    axs[0, 1].bar(bins_a[:-1], hist_a, color=bar_color, edgecolor=bar_color)
    axs[0, 1].set_title(f'Histograma: {title_a}')
    axs[0, 1].set_xlim([0, 256])  # Limitar o eixo X para intensidades de 0 a 255
    axs[0, 1].set_xlabel("Níveis de Cinza")
    axs[0, 1].set_ylabel("Frequência de Pixels")

    # Subplot 3: Imagem B em Escala de Cinza
    axs[1, 0].imshow(img_b, cmap='gray')  # Usar colormap 'gray'
    axs[1, 0].set_title(title_b)          # Título de B
    axs[1, 0].axis('off')                 # Remove eixos para imagem
    axs[1, 0].set_aspect(aspect='auto')   # Ajusta o aspecto
    
    # Subplot 2: Histograma de B
    axs[1, 1].bar(bins_b[:-1], hist_b, color=bar_color, edgecolor=bar_color)
    axs[1, 1].set_title(f'Histograma: {title_b}')
    axs[1, 1].set_xlim([0, 256])  # Limitar o eixo X para intensidades de 0 a 255
    axs[1, 1].set_xlabel("Níveis de Cinza")
    axs[1, 1].set_ylabel("Frequência de Pixels")
    
    # Layout
    plt.tight_layout()
    plt.show()



def barplot(arr: np.ndarray, title='', x_label='X', y_label='Y', figsize=(10, 10)):
    
    # Barplot( X: Domínio, Y: f(x) )
    plt.bar(range( len(arr) ), arr)
    
    # Adicionar títulos e rótulos
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Mostrar o gráfico
    plt.show()



def normalized_cumsum(X):
    return np.cumsum(X) / np.sum(X) 



def get_img_norm_hist_cum_sum(img):
    hist, bins = np.histogram(img.ravel(), bins=256, range=(0, 256))
    return normalized_cumsum(hist)



def search_nearest(X: np.uint8, y: np.uint8) -> int:

    left, right = 0, len(X) - 1
    
    while left <= right:
        mid = (left + right) // 2

        # Limita borda esquerda (Left Corner).
        mid_l = mid - 1 if mid > 0 else 0
        
        # Idéia Geral: Verifica se o elemento do meio é o alvo.
        if (X[mid_l] < y) and (y <= X[mid]):

            # Distâncias dos limites à esquerda e direta.
            dist_l = y - X[mid_l]
            dist_r = X[mid] - y

            # Retorna o índice do valor mais perto de y.
            return mid_l if dist_l < dist_r else mid
            
        # Se o alvo é maior, ignora a metade esquerda.
        elif X[mid] < y:
            left = mid + 1
            
        # Se o alvo é menor, ignora a metade direita.
        else:
            right = mid - 1

    # Distâncias dos limites à esquerda e direta.
    dist_l = y - X[mid_l]
    dist_r = X[mid] - y

    # Retorna o índice do valor mais perto de y.
    return mid_l if dist_l < dist_r else mid


# def find_lines














    