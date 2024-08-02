"""
4. Aplicar a equalização local de histograma conforme apresentado
   no slide 50 da mesma aula.
"""

import sys
from typing import Callable
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# Aplica filtro em ndarray bidimensional (i.e. grayscale).
def apply_local_mask(arr_source: np.ndarray,
                      function: Callable[[np.ndarray], np.ndarray],
                      kernel_height: int,
                      kernel_width: int) -> np.ndarray:
    """Cria uma cópia da array original modificada pela função fornecida.
    kernel_height é a altura em píxeis da janela local.
    kernel_width é a largura em píxeis da janela local.
    Larguras e alturas pares são incrementadas em 1, devido ao cálculo dos
    offsets. Isso é desejável e proposital, pois queremos um píxel no meio
    da janela, o píxel em arr_source[y, x] para o qual estamos calculando
    seu valor na imagem resultante.
    """

    # Inicia imagem de retorno como uma cópia da original, para manter a borda.
    arr_result = arr_source.copy()

    # Obtém as dimensões da array de origem.
    arr_height, arr_width = arr_result.shape

    # Calcula offsets do miolo do kernel.
    y_off = kernel_height // 2
    x_off = kernel_width // 2

    # Nessa versão da função, usamos os offsets para não processar as bordas.
    # Não é uma solução muito decente, mas serve para os fins desse trabalho.
    for y in range(y_off, arr_height - y_off):
        for x in range(x_off, arr_width - x_off):

            # Pega elementos da janela.
            kernel = arr_source[y-y_off:y+y_off+1,
                                x-x_off:x+x_off+1]

            # Aplica função fornecida.
            kernel = function(kernel)

            # Pega elemento central do kernel e salva na imagem modificada.
            arr_result[y,x] = kernel[y_off, x_off]

    # Retorna imagem modificada.
    return arr_result


# Imagem default
path_img = 'Fig0326(a)(embedded_square_noisy_512).tif'

# Se foi fornecido outra imagem, para usar com as dos slides da aula.
if len(sys.argv) > 1:
    path_img = sys.argv[1]

# Carrega imagem em escala de cinza
img_orig = cv.imread(path_img, cv.IMREAD_GRAYSCALE)

img_eqlzd = apply_local_mask(img_orig, cv.equalizeHist, 9, 9)


""" SAÍDA """

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(img_orig.ravel(), 256, (0, 256))
plt.title('Histograma da Imagem Original')

plt.subplot(1, 2, 2)
plt.hist(img_eqlzd.ravel(), 256, (0, 256))
plt.title('Histograma da Imagem Equalizada')

plt.show()

cv.imshow('Original', img_orig)
cv.imshow('Equalized', img_eqlzd)

cv.waitKey(0)
cv.destroyAllWindows()
