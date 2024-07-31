"""
1. Bit slicing.
(A) Visualizar as 8 camadas binárias de uma imagem uint8.
(B) Visualizar a composição das camadas [b_7, ..., b_0] zerando as camadas; b_0; b_0 e b_1; b_0, b_1 e b_2 e assim sucessivamente.

2. Aplicar a equalização de histograma mostrando os histogramas inicial e final das imagens da aulas ch03-transformacao de intensidade (pollen, cameraman, etc).

3. Aplicar a correspondência de histograma (matching) na imagem do slide 45 (moon).

4. Aplicar a equalização local de histograma conforme apresentado no slide 50 da mesma aula.
"""


import os
import numpy as np
import cv2

# Configurar o NumPy para mostrar mais elementos
np.set_printoptions(threshold=1000, edgeitems=16, linewidth=143)

img_path = 'dazai.png'

img_orig = cv2.imread(img_path, cv2.IMREAD_REDUCED_GRAYSCALE_8)

height, width = image.shape

x1 = width // 4
y1 = 0
x2 = width // 2
y2 = height // 4

img_part = img_orig[y1:y2, x1:x2]

print( np.array2string(img_orig, precision=1, separator=' ', suppress_small=True) )
cv2.imshow('Original', img_orig)

cv2.waitKey(0)
cv2.destroyAllWindows()
