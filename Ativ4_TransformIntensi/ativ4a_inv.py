
import cv2
from support import invert_image

img_path = 'Osamu-Dazai-Bungo-Stray-Dogs.png'

img_src_gray = cv2.imread(img_path, cv2.IMREAD_REDUCED_GRAYSCALE_2)
img_src_color = cv2.imread(img_path, cv2.IMREAD_REDUCED_COLOR_2)

img_gray_inv = invert_image(img_src_gray)
img_color_inv = invert_image(img_src_color)

cv2.imshow('Negative Gray', img_gray_inv)
cv2.imshow('Gray', img_src_gray)

cv2.imshow('Negative Color', img_color_inv)
cv2.imshow('Color', img_src_color)

cv2.waitKey(0)
cv2.destroyAllWindows()
