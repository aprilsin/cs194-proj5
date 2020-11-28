import itertools

import numpy as np
import skimage.transform

import constants
import filters
import homography
import utils


def refine_matches():
    pass


def get_patches(img, corners) -> np.ndarray:
    """make all detected corner an 8x8 grid"""
    img = filters.gauss_blur(img)
    img = utils.to_gray(img)

    patches = []
    for (r, c) in corners:

        patch = img[r - 20 : r + 20, c - 20 : c + 20]  # get a 40x40 patch
        patch = filters.gauss_blur(patch)
        assert patch.shape == (40, 40), f"{patch.shape}, {(r, c) = }, {img.shape = }"

        patch = patch[::5, ::5]  # downsample by taking every 5th pixel
        assert patch.shape == (8, 8), patch.shape

        patch = utils.normalize(patch)

        patches.append(patch)

    return patches
