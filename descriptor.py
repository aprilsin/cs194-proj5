import itertools

import numpy as np
import skimage.transform

import filters
import homography
import utils
from constants import *


def vectorize(patch):
    assert patch.shape == (8, 8), patch.shape
    return np.reshape(patch, (1, 64))


def normalize(mat):
    return (mat - np.mean(mat)) / np.std(mat)


def refine_matches():
    pass


def get_patches(img, corners) -> np.ndarray:
    """make all detected corner an 8x8 grid"""
    img = filters.gauss_blur(img)
    img = utils.to_gray(img)  # TODO make it work with colored patches

    patches = []
    for (r, c) in corners:
        patch = img[r - 20 : r + 20, c - 20 : c + 20]  # get a 40x40 patch

        # patch = skimage.transform.resize(patch, (8, 8))  # downsample
        patch = patch[::5, ::5, :]  # downsample by taking every 5th pixel

        patch = normalize(patch)  # normalize

        patches.append(patch)

    return np.array(patches)


def dist_patches(patch1, patch2):
    """
    patch1 and patch2 are 8x8 grids.
    """
    assert patch1.shape == patch2.shape == (8, 8), (patch1.shape, patch2.shape)
    patch1 = np.reshape(patch1, (1, 64))
    patch2 = np.reshape(patch2, (1, 64))
    return np.sum((patch1 - patch2) ** 2)  # sum squared distance
