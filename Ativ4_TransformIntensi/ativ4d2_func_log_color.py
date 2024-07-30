import cv2
import numpy as np


def apply_log_transform(image):
    # Normalizar a imagem para valores entre 0 e 1
    normalized_image = image / 255.0

    # Aplicar a transformação logarítmica
    c = 1  # constante de escala, pode ser ajustada conforme necessário
    log_image = c * np.log1p(normalized_image)  # log1p(x) é log(1 + x)

    # Escalar de volta para valores entre 0 e 255
    log_image = cv2.normalize(log_image, None, 0, 255, cv2.NORM_MINMAX)

    # Converter para tipo de dado adequado
    log_image = np.uint8(log_image)

    return log_image


img_path = 'Osamu-Dazai-Bungo-Stray-Dogs.png'

# Carregar a imagem colorida
image = cv2.imread(img_path, cv2.IMREAD_REDUCED_COLOR_2)

# Verificar se a imagem foi carregada corretamente
if image is None:
    raise ValueError("Imagem não encontrada.")

# Aplicar a transformação logarítmica em cada canal de cor
log_image = apply_log_transform(image)

# Exibir a imagem original e a imagem com transformação logarítmica
cv2.imshow('Imagem Original', image)
cv2.imshow('Imagem Logaritmica', log_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Opcional: salvar a imagem resultante
# cv2.imwrite('dazai_logarithmic_color.jpg', log_image)
