
import os

import numpy as np
import requests
import cv2


# Procedimento para baixar e salvar a imagem caso ela não esteja presente no diretório atual.
def download_image(url, save_path):
    if not os.path.exists(save_path):
        print('Obtendo arquivo de', url)
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print('Arquivo baixado e salvo em', save_path)
        else:
            raise Exception(f'ERRO: Problema ao baixar imagem:\n{response.status_code}')
    else:
        print('Arquivo já está presente em', save_path)


# Envelope com mensagem de erro caso o arquivo não exista.
def load_image(filepath, mode=cv2.IMREAD_COLOR):
    if os.path.exists(filepath):
        return cv2.imread(filepath, mode)
    else:
        raise Exception(f'Arquivo não encontrado em {filepath}')


# Prepara intervalos de quantização.
def quantize_steps(min, max, q):
    step = (max - min) // q
    steps = []
    for i in range(0, q):
        steps.append(min + i*step)
    return steps


url_img = 'https://github.com/tfvieira/digital-image-processing/raw/main/img/ctskull.tif'  # URL da imagem
path_file = 'ctskull.tif'  # Local da imagem

# Baixar a imagem se não existir localmente
download_image(url_img, path_file)

# Carregamento da imagem.
img_orig = load_image(path_file)                        # Normal/default, colorido.
img_gray = load_image(path_file, cv2.IMREAD_GRAYSCALE)  # Escala de cinza.

# Ver a diferença nas ndarrays
print('Dimensões da ndarray com a imagem original: ', img_orig.shape)
print('Dimensões da ndarray com a imagem grayscale:', img_gray.shape)

cv2.imshow('Original Image', img_orig)   # Ver imagem original
cv2.imshow('Grayscale Image', img_gray)  # Ver imagem em escala de cinza

# Quantiza em 2 níveis.
# img_g2 = np.where(img_gray > 126, 1, 0)
# Converte para escala de 8 bits, cv2.imshow() não funciona sem esse ajuste.
# img_g2 = (img_g2 * 255).astype(np.uint8)

# Quantiza em dois níveis, já ajustado para 8 bits.
img_g2 = np.where(img_gray > 126, 255, 0).astype(np.uint8)

print('Dim img_g2:', img_g2.shape)
print(img_g2)
cv2.imshow('N2', img_g2)

cv2.waitKey(0)
cv2.destroyAllWindows()
