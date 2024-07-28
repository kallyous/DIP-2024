import cv2
import numpy as np


# Carregar imagens
img_dazai = cv2.imread('Osamu-Dazai-Bungo-Stray-Dogs.png')
img_mask_circle = cv2.imread('Circle-Mask.png')

cv2.imshow('Dazai', img_dazai)
cv2.imshow('Mascara Circular', img_mask_circle)
cv2.waitKey(0)


# 1. Crop (Recorte)
def cropImage(image, start_row, start_col, end_row, end_col):
    return image[start_row:end_row, start_col:end_col]


# Exemplo de recorte
img_dazai_cropped = cropImage(img_dazai, 50, 50, 200, 200)
cv2.imshow('Dazai Recortado', img_dazai_cropped)
cv2.waitKey(0)


# 2. Resize (Redimensionamento)
def resizeImage(image, width, height):
    return cv2.resize(image, (width, height))


# Exemplo de redimensionamento
img_dazai_resized = resizeImage(img_dazai, 300, 300)
cv2.imshow('Dazai Escalonado', img_dazai_resized)
cv2.waitKey(0)


# Ajustar as duas imagens para o mesmo tamanho para as operações seguintes
img_dazai_300 = resizeImage(img_dazai, 300, 300)
img_mask_circle_300 = resizeImage(img_mask_circle, 300, 300)

# 3. Adição
img_added = cv2.add(img_dazai_300, img_mask_circle_300)
cv2.imshow('Adicao das Imagens', img_added)
cv2.waitKey(0)

# 4. Subtração
img_subtracted = cv2.subtract(img_dazai_300, img_mask_circle_300)
cv2.imshow('Subtracao das Imagens', img_subtracted)
cv2.waitKey(0)

# 5. Multiplicação
img_multiplied = cv2.multiply(img_dazai_300, img_mask_circle_300)
cv2.imshow('Multiplicacao das Imagens', img_multiplied)
cv2.waitKey(0)


# 5.1. Mascaramento
def maskImage(image, mask):
    return cv2.bitwise_and(image, mask)


img_dazai_masked = maskImage(img_dazai_300, img_mask_circle_300)
cv2.imshow('Dazai com Mascara Circular', img_dazai_masked)
cv2.waitKey(0)


# 5.2. Blending (Mistura)
def blendImages(image1, image2, alpha=0.5, beta=0.5, gamma=0):
    return cv2.addWeighted(image1, alpha, image2, beta, gamma)


# Exemplo de blending
img_blended = blendImages(img_dazai_300, img_mask_circle_300, alpha=0.7, beta=0.3)
cv2.imshow('Mescla das Imagens', img_blended)
cv2.waitKey(0)


cv2.destroyAllWindows()
