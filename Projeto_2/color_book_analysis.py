import os.path

import numpy as np
import cv2 as cv
import skimage as ski

from utils import *


# Configurar o estilo de fundo escuro
plt.style.use('dark_background')

fact = 2
width = 800 // fact
height = 600 // fact
shape = (height, width)

path_color_book = "Book"
path_results = os.path.join(path_color_book, "Analysis")



# Faz diretório de resultados caso não exista.
if not os.path.exists(path_results):
    os.makedirs(path_results)


# Localizar máscaras das cores do livro/catálogo.
color_mask_files = book_masks_load(path_color_book)
for mask in color_mask_files:
    print(mask["name"], "->", mask["path"])


# Carregar catálogo de cores.
path_img_color_book = os.path.join(path_color_book, "Book_01_Clean.png")
img_book = cv.imread(path_img_color_book, cv.IMREAD_COLOR)


# Bagaceira com cada cor.
colors_data = []
for color_mask in color_mask_files:

    # Carrega máscara e aplica no catálogo, obtendo RdI da cor desejada.
    mask = cv.imread(color_mask["path"], cv.IMREAD_GRAYSCALE)
    img = cv.bitwise_and(img_book, img_book, mask=mask)
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Encontra as coordenadas do menor retângulo contendo a máscara.
    ys, xs = np.where(mask == 255)        # Vai retornar as coordenadas dos pixels brancos, em duas listas.
    x_min, x_max = xs.min(), xs.max() +1  # O fatiamento que vem a seguir, é feito com intervalos inclusivos
    y_min, y_max = ys.min(), ys.max() +1  # à esquerda e exclusivos à direita, então adicionamos +1 às direitas.

    # Recorta a imagem usando as coordenadas obtidas.
    img = img[y_min:y_max, x_min:x_max]

    # Recorta a máscara pro mesmo tamanho, já que as coordenadas são literalmente as mesmas.
    mask = mask[y_min:y_max, x_min:x_max]

    # Histogramas dos canais, somente na região das máscaras.
    hist_hue = cv.calcHist([img[:,:,0]], channels=[0], mask=mask, histSize=[256], ranges=[0, 256])
    hist_sat = cv.calcHist([img[:, :, 1]], channels=[0], mask=mask, histSize=[256], ranges=[0, 256])
    hist_val = cv.calcHist([img[:, :, 2]], channels=[0], mask=mask, histSize=[256], ranges=[0, 256])

    # Organiza informações em dict.
    data = {
        "name": color_mask["name"],
        "src": color_mask["path"],
        "img": img,
        "mask": mask,
        "hist_hue": hist_hue,
        "hist_sat": hist_sat,
        "hist_val": hist_val
    }

    # Guarda informação dessa cor.
    colors_data.append(data)


for color in colors_data:

    # Base do caminho e nome para as imagens a serem salvas.
    base_path = os.path.join( path_results, os.path.basename(color["src"]).split(".")[0] )
    base_path = base_path.replace("Book_Mask_", "")

    # Exporta a imagem base recortada pra RdI.
    filepath = f"{base_path}.png"
    cv.imwrite(filepath, cv.cvtColor(color["img"], cv.COLOR_HSV2BGR))

    # Exporta a máscara recortada para a RdI.
    filepath = f"{base_path}_Mask.png"
    cv.imwrite(filepath, color["mask"])

    # Plota o matiz e salva imagem.
    plot_name = color["name"] + " - Matiz"
    filepath = f"{base_path}_Hist_Hue.png"
    plot_cv_hist(color["hist_hue"], title=plot_name, save=True, save_path=filepath)

    # Plota a saturação e salva imagem.
    plot_name = color["name"] + " - Saturação"
    filepath = f"{base_path}_Hist_Sat.png"
    plot_cv_hist(color["hist_sat"], title=plot_name, color="white", save=True, save_path=filepath)

    # Plota tons e salva imagem.
    plot_name = color["name"] + " - Tons"
    filepath = f"{base_path}_Hist_Val.png"
    plot_cv_hist(color["hist_val"], title=plot_name, color="white", save=True, save_path=filepath)


exit(0)





































