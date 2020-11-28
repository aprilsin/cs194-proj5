import itertools
import sys
from collections import Counter
from dataclasses import dataclass
from functools import total_ordering
from queue import PriorityQueue

import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
from numpy.core.fromnumeric import argmin
from scipy.spatial.distance import pdist, squareform
from skimage.feature import corner_harris, corner_peaks, peak_local_max
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate

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
    return h, coords  # r, c = y, x


def anms(strength, detected_coords, robust_factor=0.9):
    """Returns top NUM_KEEP points from detected_coords. Everything in this
    function works with indices to detected_coords.

    detected_coords: shape = (P,2) where P := number of pts
    """
    # P,P where entries are *distances* between detected_coords[i] and detected_coords[j]
    dists = squareform(pdist(detected_coords))
    print(f"{len(detected_coords)=}")

    robust_strength = (
        robust_factor * strength[detected_coords[:, 0], detected_coords[:, 1]]
    )  # shape = (P, )

    # candidates = set()
    #
    # for r in trange(constants.MAX_RADIUS, constants.MIN_RADIUS, -constants.MIN_RADIUS):
    #     dists_mask = dists > r
    #
    #     for (i, p) in enumerate(tqdm(detected_coords)):
    #
    #         # nonzero: j such that d(detected_coords[i],detected_coords[j])>r
    #         dist_js = np.argwhere(dists_mask[i])
    #         strength_js = np.argwhere((strength[p[0], p[1]] <= robust_strength)).ravel()
    #
    #         common_js = np.intersect1d(dist_js, strength_js, assume_unique=True)
    #         # to get pts, map back js
    #         for j in common_js:
    #             if len(candidates) == constants.NUM_KEEP:
    #                 return np.array(list(candidates))
    #             candidates.add(tuple(detected_coords[j]))
    # print(len(candidates))

    N = len(detected_coords)
    # second slot is for best j for that i
    R = [[float("inf"), None] for _ in range(N)]

    # for i, p in enumerate(tqdm(detected_coords)):
    #     best_j = None
    #     for j, p_ in enumerate(tqdm(detected_coords)):
    #         if strength[p] < robust_strength[p_] and dists[i, j] < R[i][0]:
    #             R[i][0] = dists[i, j]
    #             best_j = j
    #     R[i][1] = best_j

    for i, p in enumerate(tqdm(detected_coords)):
        larger_mag_idxs = np.argwhere(strength[p[0], p[1]] < robust_strength).ravel()
        # pick the index that's closest
        try:
            best_idx = min(larger_mag_idxs, key=lambda j: dists[i, j])
        except ValueError:
            # found local max, skip it
            continue
        R[i] = [dists[i, best_idx], best_idx]

    R = sorted(R, key=lambda x: x[0], reverse=True)
    # skip any values that are inf distance (global maxima)
    candidate_idxs = {idx for dist, idx in R if idx is not None and np.isfinite(dist)}
    # make a set since we want uniq ones and multiple points can have the same
    # pt as nearest local max

    candidate_idxs = np.array(list(candidate_idxs))
    candidates = detected_coords[candidate_idxs]

    utils.assert_coords(candidates)
    # return best_corners
    return candidates
