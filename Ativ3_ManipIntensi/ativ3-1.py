
import os
import requests
import numpy as np
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


def quanti_slice(q, min=0, max=255):

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


# Calcula intervalos e quantiza x para q quantis.
def quantize(x, ruler):
    for i in range(1, len(ruler['steps'])):
        if x < ruler['steps'][i]:
            return ruler['values'][i-1]
    return ruler['values'][-1]


# Config
url_img = 'https://github.com/tfvieira/digital-image-processing/raw/main/img/ctskull.tif'
path_file = 'ctskull.tif'

# Carregamento
download_image(url_img, path_file)
img_src = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)

# Imagem fonte
print('Dim src:', img_src.shape)
cv2.imshow('Src Img', img_src)

'''
ruler = quanti_slice(2)
print('Régua de Quantização:')
print('  Passos  ', ruler['steps'])
print('  Valores ', ruler['values'])
print()

print('-1', quantize(-1, ruler))
print('0', quantize(0, ruler))
print('1', quantize(1, ruler))

print('63', quantize(63, ruler))
print('64', quantize(64, ruler))
print('65', quantize(65, ruler))

print('127', quantize(127, ruler))
print('128', quantize(128, ruler))
print('129', quantize(129, ruler))

print('169', quantize(169, ruler))
print('170', quantize(170, ruler))
print('171', quantize(171, ruler))

print('254', quantize(254, ruler))
print('255', quantize(255, ruler))
print('256', quantize(256, ruler))
'''

for q in (128, 64, 32, 16, 8, 4, 2):
    ruler = quanti_slice(q)
    img_quantified = np.vectorize(lambda x: quantize(x, ruler))(img_src)
    img_quantified = img_quantified.astype(np.uint8)
    cv2.imshow(f'Q{q}', img_quantified)

cv2.waitKey(0)
cv2.destroyAllWindows()
