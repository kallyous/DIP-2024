
import cv2


# Config
path_img_1 = 'Ativ3-2_frame-1.jpg'
path_img_2 = 'Ativ3-2_frame-2.jpg'

# Carregamento
img_1 = cv2.imread(path_img_1, cv2.IMREAD_REDUCED_GRAYSCALE_2)
img_2 = cv2.imread(path_img_2, cv2.IMREAD_REDUCED_GRAYSCALE_2)

# Exibe imagens
print('Dim 1:', img_1.shape)
print('Dim 2:', img_2.shape)
cv2.imshow('Frame 1', img_1)
cv2.imshow('Frame 2', img_2)

# Calcula diferença entre as iamgens.
img_sub = img_2 - img_1
print('Dim 3:', img_1.shape)
cv2.imshow('Diff', img_sub)

# Obtém imagem borrada para redução de ruído.
img_blurr = cv2.GaussianBlur(img_sub, (9, 9), 0)
cv2.imshow('Blurr', img_blurr)

# Passa por um limiar para remover pixels de ruído.
_, img_threshold = cv2.threshold(img_blurr, 200, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold', img_threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()
