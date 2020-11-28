import itertools
import sys
from dataclasses import dataclass
from functools import total_ordering
from queue import PriorityQueue
from tqdm import trange
from tqdm.contrib import tenumerate

import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
from scipy.spatial.distance import pdist, squareform
from skimage.feature import corner_harris, corner_peaks, peak_local_max

# import constants
import constants
import filters
import homography
import utils


def get_harris(im, edge_discard=20) -> list:
    """Returns a list of harris corners of img as Corner objects. Harris
    corners near the edge are discarded and the coordinates of the remaining
    corners are returned.

    Args:
    im -- b&w image
    edge_discard (optional) --  amount to discard on the edge (default 5 pixels)

    Output:
    corners -- a list of Corner instances
    """
    assert im.ndim == 2, im.shape
    assert edge_discard >= 20
    im = filters.gauss_blur(im)

    # find harris corners
    h = corner_harris(im, method="eps", sigma=1)
    coords = corner_peaks(
        h,
        min_distance=constants.MIN_RADIUS,
        indices=True,
        threshold_rel=constants.HARRIS_STRENGTH_THRESHOLD,
    )

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (
        (coords[:, 0] > edge)
        & (coords[:, 0] < im.shape[0] - edge)
        & (coords[:, 1] > edge)
        & (coords[:, 1] < im.shape[1] - edge)
    )
    coords = coords[mask]

    # return h, coords
    utils.assert_coords(coords)
    # return h, np.flip(coords, axis=-1)  # to get (x, y)
    return h, coords


def anms(strength, detected_coords, robust_factor=0.9):
    """Everything in this function works with indices to detected_coords.
    detected_coords: shape = (P,2) where P := number of pts

    Returns top NUM_KEEP points from detected_coords.
    """
    # P,P where entries are *distances* between detected_coords[i] and detected_coords[j]
    dists = squareform(pdist(detected_coords))
    # only care about common pts
    # shape = P,
    robust_strength = (
        robust_factor * strength[detected_coords[:, 0], detected_coords[:, 1]]
    )
    candidates = set()

    for r in range(constants.MAX_RADIUS, constants.MIN_RADIUS, -constants.MIN_RADIUS):
        dists_mask = dists > r

        for (i, p) in enumerate(detected_coords):

            # nonzero: j such that d(detected_coords[i],detected_coords[j])>r
            dist_js = np.argwhere(dists_mask[i])
            strength_js = np.argwhere((strength[p[0],p[1]] <= robust_strength)).ravel()

            common_js = np.intersect1d(dist_js, strength_js, assume_unique=True)
            # to get pts, map back js
            candidates.update(tuple(x) for x in detected_coords[common_js])

            for c in candidates:
                if len(candidates) == constants.NUM_KEEP:
                    return np.array(list(candidates))
                candidates.add(c)
    return np.array(list(candidates))
