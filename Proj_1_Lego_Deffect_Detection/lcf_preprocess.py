import argparse
import copy
from pathlib import Path

import cv2 as cv
import numpy as np

from lcf_utils import get_img_norm_hist_cum_sum
from lcf_utils import search_nearest



def load_image_gray(image_path: Path) -> np.ndarray:
    """
    Load an image from a file path and convert it to grayscale.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        np.ndarray: Grayscale image.
    """
    
    return cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)



def contrast_boost(image_set: list) -> list:
    img_out_set = []
    for img in image_set:
        img_out_set.append( cv.equalizeHist(img) )
    return img_out_set



def find_binary_divide(img: np.ndarray, blk=0.5) -> np.uint8:

    nhcdf = get_img_norm_hist_cum_sum(img)
    d = search_nearest(nhcdf, blk)
    
    return d



def binarize(image_set: list, divide=128) -> list:
    img_out_set = []
    for img in image_set:
        img_out_set.append( np.where(img < divide, 0, 1) )
    return img_out_set



def preprop(image_set: list) -> list:

    img_out_set = copy.deepcopy(image_set)

    img_out_set = contrast_boost(img_out_set)
    img_out_set = binarize(img_out_set)

    return img_out_set
