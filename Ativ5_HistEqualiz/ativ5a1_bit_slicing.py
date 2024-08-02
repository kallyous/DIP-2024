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
mask_b0 = 0b00000001
mask_b1 = 0b00000010
mask_b2 = 0b00000100
mask_b3 = 0b00001000
mask_b4 = 0b00010000
mask_b5 = 0b00100000
mask_b6 = 0b01000000
mask_b7 = 0b10000000

# Pra tacar tudo num laço depois.
bmasks = [mask_b0, mask_b1, mask_b2, mask_b3,
          mask_b4, mask_b5, mask_b6, mask_b7]

# Carregamento da entrada.
img_orig = cv2.imread(img_path, cv2.IMREAD_REDUCED_GRAYSCALE_8)
cv2.imshow('Original', img_orig)

# Pega dimensões da iamgem. Como é cinza, vai ser só altura e largura.
height, width = img_orig.shape

""" Essa escolha foi arbitrária, pra ter bastante variação de valores nos
píxeis, e dar uma boa vizualização dos resultados. """
x1 = int(width // 3) - 32
y1 = int(height // 3) - 16
x2 = int(width // 3) * 2 - 32
y2 = int(height // 3) * 2 - 16

# Recorta para região escolhida.
img_part = img_orig[y1:y2, x1:x2]
# Imagens originais e corte.
cv2.imshow('Slice', img_part)

# Exibe array da região escolhida.
print('Original:')
print(np.array2string(img_part,
                      precision=1,
                      separator=' ',
                      suppress_small=True))

# Laço pra exibir todas as camadas.
i = 0
for mask in bmasks:
    img_mask = (img_part & mask) >> i
    print(f'\nCamada b{i}:')
    print(np.array2string(
                img_mask,
                precision=1,
                separator=' ',
                suppress_small=True))
    i += 1


""" Dá a opção de encerrar ou não.
Isso está aqui apenas pois cansei de fechar sem querer as janelas abertas
pelo OpenCV, ao dar Alt+Tab para olhar o que codei e comparar os resultados
esperados com o que realmente aconteceu na a saída... """
choice = input('Fechar janelas [S/n] ? ')
if choice.lower() in ['n', 'no', 'nao', 'não']:
    cv2.waitKey(0)
cv2.destroyAllWindows()
