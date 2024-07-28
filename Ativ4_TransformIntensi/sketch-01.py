
import os
import numpy as np
import cv2
import math


""" PROCEDIMENTOS """

def inverse(x):
    """Inverte um pixel."""
    return 255 - x


def gamma_correction(value, gamma):
    """
    Aplica correção gamma a um único valor de intensidade de pixel.

    Parameters:
    value (int): Intensidade do pixel (0-255).
    gamma (float): Valor gamma para correção.

    Returns:
    int: Novo valor de intensidade após correção gamma.
    """

    # Normaliza o valor do pixel para [0, 1]
    normalized = value / 255.0

    # Aplica a correção gamma
    corrected = normalized ** gamma

    # Converte de volta para a escala [0, 255]
    return corrected * 255.0


""" CONFIGURAÇÕES """

img_path = 'Osamu-Dazai-Bungo-Stray-Dogs.png'
gamma_value = 0.5


""" ENTRADA """

img_src = cv2.imread(img_path, cv2.IMREAD_REDUCED_COLOR_2)
cv2.imshow('Original', img_src)
cv2.waitKey(0)


""" INVERSA """

# Vetorizar função de inversão para ser aplicada em uma array numpy multidimensional.
vectorized_inverse = np.vectorize(inverse)

# Aplica função em todos so canais de todos dos pixeis da imagem.
img_inverted = vectorized_inverse(img_src)

# Exibe resultado.
cv2.imshow('Inversao / Negativa', img_inverted)

# Aguarda tecla qualquer para continuar script.
cv2.waitKey(0)


""" GAMMA CORRECTION """

# Vetorizar função de correção gamma para ser aplicada em uma array numpy multidimensional.
vectorized_gamma_correction = np.vectorize(gamma_correction)

'''Cria uma numpy ndarray com mesmas dimensões que a imagem original, inicializada com zeros,
para armazenar os canais sendo processados por vectorized_gamma_correction.'''
# TODO: Ainda não entendi porque isso  é necessário com a função de correção gamma, mas não para a inversão.
img_gamma_corrected = np.zeros_like(img_src)

# Laço que aplica vectorized_gamma_correction() a cada canal de img_src e armazena o resultado em img_gamma_corrected.
for i in range(3):  # Para cada canal de cor (B, G, R)
    img_gamma_corrected[:, :, i] = vectorized_gamma_correction(img_src[:, :, i], gamma_value)

# Exibe resultado.
cv2.imshow('Correcao Gamma', img_gamma_corrected)

# Aguarda tecla qualquer para continuar script.
cv2.waitKey(0)


""" FIM """

# Fecha todas as janelas abertas.
cv2.destroyAllWindows()
