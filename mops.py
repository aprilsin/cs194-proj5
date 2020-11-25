import itertools

import numpy as np

import filters
import harris
import utils

def detect_corners(img):
    # num_levels = 5
    # g_stack = filters.gaussian_stack(img)
    # for level in g_stack:
    assert img.ndim == 3, img.shape
    blurred = filters.gauss_blur(img)
    harris_stength, coords = harris.get_corners(utils.to_gray(blurred))
    return coords


def match_features(im1_grids, im2_grids):
    matched = []
    for i, j in itertools.product(len(im1_grids), len(im2_grids)):
        if np.equals(im1_grids[i], im2_grids[i]):
            matched.append([i, j])
    return matched


def get_patches(img, corners):
    """make all detected corner an 8x8 grid"""
    patches = []
    for (r, c) in corners:
        patch = img[r-20:r+20, c-20:c+20]
        # downsample
        patch = skimage.transform.resize(patch, (8, 8))
        # normalize
        patch -= mean(patch)
        patch /= np.std(patch)
        patches.append(patch)
    return patches
