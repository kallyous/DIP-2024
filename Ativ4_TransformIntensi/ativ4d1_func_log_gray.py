import cv2
import numpy as np
from support import cropImage


# Carregar a imagem em escala de cinza
img_src = cv2.imread('Fig0305(a)(DFT_no_log).tif', cv2.IMREAD_GRAYSCALE)

# Verificar se a imagem foi carregada corretamente
if img_src is None:
    raise ValueError("Imagem não encontrada.")

# Normalizar a imagem para valores entre 0 e 1
img_norm = img_src / 255.0

# Aplicar a transformação logarítmica
# TODO: não funcionou tão bem quanto esperado, estudar porquê e atualizar aqui.
c = 1  # constante de escala, pode ser ajustada conforme necessário
img_log = c * np.log1p(img_norm)  # log1p(x) é log(1 + x)
'''np.log1p() é usada para calcular o logaritmo natural de (1 + x),
o que ajuda a evitar problemas com log(0).'''

# Escalar de volta para valores entre 0 e 255
img_log = cv2.normalize(img_log, None, 0, 255, cv2.NORM_MINMAX)

# Converter para tipo de dado adequado
img_log = np.uint8(img_log)

# Exibir a imagem original e a imagem com transformação logarítmica
cv2.imshow('Imagem Original', img_src)
cv2.imshow('Imagem Logaritmica', img_log)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Opcional: salvar a imagem resultante
# cv2.imwrite('dazai_logarithmic.jpg', log_image)
