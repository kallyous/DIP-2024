
import numpy as np
import cv2
import matplotlib.pyplot as plt


def multiply(x, b, min, max):
    y = x * b
    if y < min:
        return min
    if y > max:
        return max
    return int(y)


# Configurar o NumPy para mostrar mais elementos
np.set_printoptions(threshold=1000, edgeitems=22, linewidth=280)

# Imagem de entrada.
path_img_src = 'Osamu-Dazai-Bungo-Stray-Dogs.png'

# Região 1
r1_start = 0
r1_end = 50

# Reginão 2
r2_start = 160
r2_end = 185

img_src = cv2.imread(path_img_src, cv2.IMREAD_REDUCED_GRAYSCALE_4)
cv2.imshow('Original', img_src)
# print(img_src)

# Vai receber imagem de saída.
img_out = img_src.copy()
# img_out = np.zeros_like(img_src)

# Vai receber as máscaras que filtram quais pixeis passarão pela função.
mask = (img_src >= r1_start) & (img_src < r1_end)  # Boleanos

# Vetoriza e aplica função de multiplicação nas células mascaradas.
img_out[mask] = np.vectorize(
    lambda x: multiply(x, 1.25, 0, 255))(img_src[mask])

cv2.imshow('Output 1', img_out)

# Atualiza máscara para saber quais pixeis serão copiados sem alteração.
# mask = (img_src >= r1_end) & (img_src < r2_start)
# Copia os píxeis inalterados.
# img_out[mask] = img_src[mask]
# cv2.imshow('Output 2', img_out)

# Atualiza máscara para saber quais píxeis passarão pela função 2.
mask = (img_src >= r2_start) & (img_src < r2_end)

# Vetoriza e aplica função 2 nos píxeis especificados.
img_out[mask] = np.vectorize(
    lambda x: multiply(x, 0.75, 0, 255))(img_src[mask])

cv2.imshow('Output 3', img_out)


# Calcula histograma da imagem original
hist = cv2.calcHist([img_src], [0], None, [256], [0, 256])

# Plotar o histograma
plt.figure()
plt.title("Histograma da imagem original")
plt.xlabel("Intensidade de Pixel")
plt.ylabel("Frequência")
plt.plot(hist)
plt.xlim([0, 256])
plt.savefig('dazai-histogram-orig.jpg')

# Calcula histograma da imagem alterada
hist = cv2.calcHist([img_out], [0], None, [256], [0, 256])

# Plotar o histograma
plt.figure()
plt.title("Histograma da imagem alterada")
plt.xlabel("Intensidade de Pixel")
plt.ylabel("Frequência")
plt.plot(hist)
plt.xlim([0, 256])
plt.savefig('dazai-histogram-mod.jpg')

cv2.waitKey(0)
cv2.destroyAllWindows()
