"""
1. Bit slicing.
    (A) Visualizar as 8 camadas binárias de uma imagem uint8.
    (B) Visualizar a composição das camadas [b_7, ..., b_0] zerando as camadas;
        b_0; b_0 e b_1; b_0, b_1 e b_2 e assim sucessivamente.
"""

import numpy as np
import cv2


""" Configurar o NumPy para mostrar mais elementos.
Essa escolha é arbitrária, baseada no monitor que uso no lab, um
ultra-whide screen de 2560x1080 pixels. """
np.set_printoptions(threshold=1500, edgeitems=22, linewidth=280)

# Imagem de entrada.
img_path = 'dazai.png'

# Serão usados nas máscaras binárias.
mask_b0 = 0b11111110
mask_b1 = 0b11111101
mask_b2 = 0b11111011
mask_b3 = 0b11110111
mask_b4 = 0b11101111
mask_b5 = 0b11011111
mask_b6 = 0b10111111
mask_b7 = 0b01111111

# Pra tacar tudo num laço depois.
bmasks = [mask_b0, mask_b1, mask_b2, mask_b3,
          mask_b4, mask_b5, mask_b6, mask_b7]

# Carregamento da entrada.
img_orig = cv2.imread(img_path, cv2.IMREAD_REDUCED_GRAYSCALE_4)
cv2.imshow('Original', img_orig)

print('Iniciando exibição das modificações incrementais nas camadas:')

i = 0
img_compr = img_orig.copy()
for mask in bmasks:

    # Zera camada atual.
    img_compr = img_compr & mask

    # Exibe imagem com camada zerada.
    print(f'Camada b{i} zerada.')

    cv2.imshow(f'Camadas 0 a {i} zeradas', img_compr)

    # Aguarda entrada do usuário antes de passar para a próxima.
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    i += 1
