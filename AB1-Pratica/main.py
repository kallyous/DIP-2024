import os

import numpy as np
import requests
import cv2


q1a_img_url = "https://raw.githubusercontent.com/tfvieira/digital-image-processing/main/img/baboon.png"
q1a_img_path = "baboon.png"

q1b_img_url = "https://raw.githubusercontent.com/tfvieira/digital-image-processing/main/img/spectrum.tif"
q1b_img_path = "spectrum.tiff"


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


# Envelope pra mascaras
def mask_img(image, mask):
    return cv2.bitwise_and(image, mask)



if __name__ == "__main__":


    # Q1 - Inicio

    download_image(q1a_img_url, q1a_img_path)
    q1_img = load_image(q1a_img_path, cv2.IMREAD_GRAYSCALE)
    print("Resolução espacial: ", q1_img.shape)
    print("Profundidade:", type(q1_img[0][0]), ", é um inteiro de 8 bits.")  # <class 'numpy.uint8'> Nos dá 8 bits de profundidade.
    cv2.imshow("Q1 Img", q1_img)

    # Q1 a) Rotação em 45º
    center_x, center_y = q1_img.shape[1] // 2, q1_img.shape[0] // 2
    angle = 45
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    q1_img_rot = cv2.warpAffine(q1_img, rotation_matrix, (q1_img.shape[1], q1_img.shape[0]))
    q1_circle = np.zeros(q1_img.shape, dtype=np.uint8)
    cv2.circle(q1_circle, (256, 256), 100, (255), 200)
    q1_img_rot = mask_img(q1_img_rot, q1_circle)
    cv2.imshow("Q1 Img Rot", q1_img_rot)

    # Q1 b) Transformação logaritmica
    download_image(q1b_img_url, q1b_img_path)
    q1b_img = load_image(q1b_img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Q1 Spectrum", q1b_img)

    # Normalizar a imagem para valores entre 0 e 1
    img_spec_norm = q1b_img / 255.0

    # Aplicar a transformação logarítmica
    c = 1
    img_spec_log = c * np.log1p(img_spec_norm)
    img_spec_log = cv2.normalize(img_spec_log, None, 0, 255, cv2.NORM_MINMAX)
    img_spec_log = np.uint8(img_spec_log)
    cv2.imshow("Q1 Spectrum Log", img_spec_log)

    cv2.waitKey(0)
    cv2.destroyAllWindows()