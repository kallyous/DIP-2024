
import os
import numpy as np
import cv2
import math


def inverse(arr):
    return 255 - arr


def gamma(x, y):
    return ((x/255)**y) * 255


# transformação logaritmica


dazai = cv2.imread('Osamu-Dazai-Bungo-Stray-Dogs.png', cv2.IMREAD_REDUCED_GRAYSCALE_2)
cv2.imshow('Input', dazai)

# imout = inverse(dazai)
imout = np.vectorize(lambda x: gamma(x, 1))(dazai)
cv2.imshow('Output', imout)

cv2.waitKey(0)
cv2.destroyAllWindows()
