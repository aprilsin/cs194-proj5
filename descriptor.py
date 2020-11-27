import itertools

import numpy as np
import skimage.transform

import filters
import homography
import utils
from constants import *


def refine_matches():
    pass


def get_patches(img, corners) -> np.ndarray:
    """make all detected corner an 8x8 grid"""
    img = filters.gauss_blur(img)
    img = utils.to_gray(img)  # TODO make it work with colored patches?

    patches = []
    print(corners[4])
    for (y, x) in corners:
        r, c = y, x
        patch = img[r - 20 : r + 20, c - 20 : c + 20]  # get a 40x40 patch
        patch = patch[::5, ::5]  # downsample by taking every 5th pixel
        patch = utils.normalize(patch)  # normalize
        patches.append(patch)

    return np.array(patches)
