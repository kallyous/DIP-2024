
import numpy as np
import cv2

from support import gamma_correct


path_img_src = 'Osamu-Dazai-Bungo-Stray-Dogs.png'
gamma_value = .5

img_src = cv2.imread(path_img_src, cv2.IMREAD_REDUCED_COLOR_2)
cv2.imshow('Original', img_src)

vctrzd_gamma_correct = np.vectorize(gamma_correct)

img_gam_corr = np.zeros_like(img_src)

for i in range(3):
    img_gam_corr[:, :, i] = vctrzd_gamma_correct(img_src[:, :, i], gamma_value)

cv2.imshow('Gamma Corrected', img_gam_corr)

cv2.waitKey(0)
cv2.destroyAllWindows()
