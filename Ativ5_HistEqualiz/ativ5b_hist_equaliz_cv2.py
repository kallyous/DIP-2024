"""
2. Equalização de Histograma.
   Aplicar a equalização de histograma mostrando os histogramas inicial e final
   das imagens da aulas ch03-transformacao de intensidade
   (pollen, cameraman, etc).
Aqui usamos a função pronta do cv2 pra comparar os resutlados.
https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
"""

import sys
import cv2
import matplotlib.pyplot as plt


# Imagem default
list_path_img = ['ex1.jpg', 'Fig0310(b)(washed_out_pollen_image).tif',
                 'Fig0323(a)(mars_moon_phobos).tif', 'dazai.png']

# Se foi fornecido outra imagem, pra usar com as dos slides da aula.
if len(sys.argv) > 1:
    list_path_img = sys.argv[1:]

for path in list_path_img:

    img_src = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_eqlzd = cv2.equalizeHist(img_src)

    cv2.imshow(f'{path} Original', img_src)
    cv2.imshow(f'{path} Equalizada', img_eqlzd)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(img_src.ravel(), 256, (0, 256))
    plt.title(f'{path}\n Histograma Original')

    plt.subplot(1, 2, 2)
    plt.hist(img_eqlzd.ravel(), 256, (0, 256))
    plt.title(f'{path}\n Histograma Equalizado')

cv2.waitKey(0)
cv2.destroyAllWindows()

plt.show()


'''
# Carregar a imagem em escala de cinza
img_orig = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)

# Aplicar a equalização de histograma
img_eqlzd = cv2.equalizeHist(img_orig)

# Mostrar a imagem original e a imagem equalizada
cv2.imshow('Original', img_orig)
cv2.imshow('Equalizada', img_eqlzd)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Plotar os histogramas
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(img_orig.ravel(), 256, (0, 256))
plt.title('Histograma da Imagem Original')

plt.subplot(1, 2, 2)
plt.hist(img_eqlzd.ravel(), 256, (0, 256))
plt.title('Histograma da Imagem Equalizada')

plt.show()
'''