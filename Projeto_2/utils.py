import os

import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt



def imsetshow_quit(named_images: list):
    for name, img in named_images:
        cv.imshow(name, img)
    cv.waitKey()
    cv.destroyAllWindows()
    exit(0)



def plot_cv_hist(cvhistogram, figsize=(10, 5), title="", color=[], save=False, save_path=None):
    """
    Plota ndarray como histograma.

    Args:
        data:  Histograma a plotar.
        figsize (tupla widith, height):  Tamanho da figura.
        title (str):  Título para o plot.
    """

    # Criar as cores em HSV e converter para RGB
    if not color:
        for x in range(256):
            hsv = np.array([[[x % 180, 255, 255]]], dtype=np.uint8)  # HSV color
            rgb = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)  # Convert to RGB
            color.append(rgb[0, 0] / 255.0)  # Normalize RGB values for matplotlib

    plt.figure(figsize=figsize)
    plt.bar(range(256), cvhistogram.flatten(), color=color)
    plt.title(title)
    plt.xlabel('Valor')
    plt.ylabel('Frequência')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



def book_masks_load(masks_folder):

    masks = [ f for f in os.listdir(masks_folder)
              if f.startswith("Book_Mask_") ]

    masks = sorted(masks)

    for i in range(len(masks)):
        path = os.path.join(masks_folder, masks[i])
        name = masks[i].split(".")[0]
        parts = name.split("_")[3:]
        name = " ".join(parts)
        masks[i] = { "name": name, "path": path }

    return masks



def load_data(data_file_path: str):
    """Carrega CSV em dataframe ou encerra se arquivo não existir."""
    try:
        df = pd.read_csv(data_file_path)
    except FileNotFoundError as e:
        print(f"{RED}ERRO: {data_file_path} não encontrado!{CLR}\n")
        exit(2)
    return df


























