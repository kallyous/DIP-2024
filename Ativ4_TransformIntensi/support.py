
import os
import requests
import numpy as np


""" ENTRADA E SAÍDA """


# Armazena se programa está rodando em modo debug.
DEBUG = False


# Printa coisas caso DEBUG = True, usando print().
def log(*args, **kwargs):
    """
    Imprime mensagens somente se DEBUG estiver definido como True.
    :param args: Argumentos posicional que são passados para a função print.
    :param kwargs: Argumentos nomeados que são passados para a função print.
    """
    if DEBUG:
        print(*args, **kwargs)


# Define global DEBUG e retorna seu valor atualizado.
def setDebug(dbg):
    global DEBUG
    DEBUG = dbg
    return DEBUG


# Baixa e salva arquivos usando requests.get().
def download(url, save_path):
    """Recebe url de arquivo para baixar e salva em save_path."""
    if not os.path.exists(save_path):
        log('Obtendo arquivo de', url)
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)
            log('Arquivo baixado e salvo em', save_path)
        else:
            raise Exception('ERRO: Problema ao baixar arquivo:'
                            f'\n{response.status_code}')
    else:
        log('Arquivo já está presente em', save_path)


""" QUANTIZAÇÃO """


# Cria régua para quantizar valores dentro de um intervalo.
def quanti_slice(q, min=0, max=255):
    """Cria régua de 'q' intervalos dentro do intervalo [min, max], e retorna
    um dict que associa os valores limites dos intervalos aos pontos/quantis
    que são mapeados. Essa régua é para ser usada
    na função quantize(x, ruler)"""

    ruler = {}

    step = 1 + (max - min) // q
    ruler['steps'] = []
    for i in range(0, q + 1):
        ruler['steps'].append(min + i * step)

    step = (max - min) // (q-1)
    ruler['values'] = []
    for i in range(0, q):
        ruler['values'].append(min + i*step)

    return ruler


# Quantiza x para a régua fornecida.
def quantize(x, ruler):
    """Calcula a qual intervalo da régua fornecida x pertence,
    e retorna o valor para o qual x é mapeado de acordo com a régua fornecida.
    A régua esperada é o resultado de quanti_slice() para o intervalo
    e quantis desejados."""
    for i in range(1, len(ruler['steps'])):
        if x < ruler['steps'][i]:
            return ruler['values'][i-1]
    return ruler['values'][-1]


""" TRANSFORMAÇÕES BÁSICAS """


# Crop / Recorte
def cropImage(image, start_row, start_col, end_row, end_col):
    return image[start_row:end_row, start_col:end_col]


# Inverse / Negativa em Unsigned Int 8 bits
def invert(x):
    return 255 - x


# Aplica invert(x) em imagem cinza.
def invert_gray(gray_img):
    vctrzd_invert = np.vectorize(invert)
    return vctrzd_invert(gray_img)


# Aplica invert(x) em imagem colorida, canal por canal.
def invert_color(color_img):
    vctrzd_invert = np.vectorize(invert)
    img_inv = np.zeros_like(color_img)
    for i in range(3):
        img_inv[:, :, i] = vctrzd_invert(color_img[:, :, i])
    return img_inv


# Aplica invert_cinza() ou invert_color(), apropriadamente.
def invert_image(image):
    shape = image.shape
    if len(shape) == 2:
        return invert_gray(image)
    if len(shape) == 3:
        return invert_color(image)
    raise Exception(f'image.shape esperado 2 ou 3, recebido {shape}')


""" GAMMA """


# Correção gamma em Unsigned Int 8 bits
def gamma_correct(x, gamma):
    """
    Aplica correção gamma a um único valor de intensidade de pixel.

    Parameters:
    x (int): Intensidade do pixel (0-255).
    gamma (float): Valor gamma para correção.

    Returns:
    int: Novo valor de intensidade após correção gamma.
    """

    # Normaliza o valor do pixel para [0, 1]
    normalized = x / 255.0

    # Aplica a correção gamma
    corrected = normalized ** gamma

    # Converte de volta para a escala [0, 255]
    return corrected * 255.0
