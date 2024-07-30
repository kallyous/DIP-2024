import cv2
import numpy as np
from support import cropImage


# Carregar a imagem em escala de cinza
image = cv2.imread('fig-3-5.png', cv2.IMREAD_GRAYSCALE)
image = cropImage(image, 0, 0, 367, 367)

# Verificar se a imagem foi carregada corretamente
if image is None:
    raise ValueError("Imagem não encontrada.")

# Normalizar a imagem para valores entre 0 e 1
normalized_image = image / 255.0

# Aplicar a transformação logarítmica
c = 1  # constante de escala, pode ser ajustada conforme necessário
log_image = c * np.log1p(normalized_image)  # log1p(x) é log(1 + x)
'''np.log1p() é usada para calcular o logaritmo natural de (1 + x),
o que ajuda a evitar problemas com log(0).'''

# Escalar de volta para valores entre 0 e 255
log_image = cv2.normalize(log_image, None, 0, 255, cv2.NORM_MINMAX)

# Converter para tipo de dado adequado
log_image = np.uint8(log_image)

# Exibir a imagem original e a imagem com transformação logarítmica
cv2.imshow('Imagem Original', image)
cv2.imshow('Imagem Logaritmica', log_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Opcional: salvar a imagem resultante
# cv2.imwrite('dazai_logarithmic.jpg', log_image)
