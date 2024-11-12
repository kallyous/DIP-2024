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

df = load_data("train.csv")

with open("fuck.txt", "w") as file:
    file.write(df.to_string())








































