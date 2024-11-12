import numpy as np
import cv2 as cv
import skimage as ski

fact = 2
width = 800 // fact
height = 600 // fact
shape = (height, width)

def show(named_images: list):
    for name, img in named_images:
        cv.imshow(name, img)
    cv.waitKey()
    cv.destroyAllWindows()
    exit(0)

# Saturação e luma máximos
sat = np.full(shape, 255, dtype=np.uint8)
val = np.full(shape, 255, dtype=np.uint8)

# Vermelho
hue = np.full(shape, 0, dtype=np.uint8)
img_hsv = np.dstack([hue, sat, val])
img_red = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)

# Amarelo
hue = np.full(shape, 30, dtype=np.uint8)
img_hsv = np.dstack([hue, sat, val])
img_ylw = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)

# Verde
hue = np.full(shape, 60, dtype=np.uint8)
img_hsv = np.dstack([hue, sat, val])
img_grn = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)

# Ciano
hue = np.full(shape, 90, dtype=np.uint8)
img_hsv = np.dstack([hue, sat, val])
img_cyn = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)

# Azul
hue = np.full(shape, 120, dtype=np.uint8)
img_hsv = np.dstack([hue, sat, val])
img_blu = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)

# Lilás
hue = np.full(shape, 150, dtype=np.uint8)
img_hsv = np.dstack([hue, sat, val])
img_mgt = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)


show([
    ("Vermelho", img_red),
    ("Amarelho", img_ylw),
    ("Verde", img_grn),
    ("Ciano", img_cyn),
    ("Azul", img_blu),
    ("Lilas", img_mgt)
])
