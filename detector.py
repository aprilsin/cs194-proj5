import itertools
import sys
from dataclasses import dataclass
from functools import total_ordering
from queue import PriorityQueue

import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
from skimage.feature import corner_harris, corner_peaks, peak_local_max

# import constants
import constants
import filters
import homography
import utils


def get_harris(im, edge_discard=20) -> list:
    """
    Returns a list of harris corners of img as Corner objects. Harris corners near the edge are discarded and the coordinates of the remaining corners are returned.

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
        min_distance=2,
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
    return h, np.flip(coords, axis=-1)  # to get (x, y)


def anms(h_strengths, coords, eps=0.9) -> list:

    # initialize
    keep = []
    candidates = [
        (coord[0], coord[1]) for coord in coords
    ]  # turn np.ndarray into a list

    # get global maximum
    strongest_corner = None
    strongest_strength = 0
    for corner in candidates:
        if h_strengths[corner] > strongest_strength:
            strongest_corner = corner
            strongest_strength = h_strengths[corner]
    keep.append(strongest_corner)
    candidates.remove(strongest_corner)

    r = constants.MIN_RADIUS  # suppression radius
    while (
        len(keep) < constants.NUM_KEEP
        and len(candidates) > 0
        and r < constants.MAX_RADIUS
    ):
        # compute ssd for all kept centers / coords
        sq_dist = utils.dist2(keep, candidates)
        print(sq_dist)
        # outlier rejection
        mask = np.where(sq_dist <= 10)
        print(sq_dist[mask])
        sq_dist = sq_dist if mask else float("inf")
        print(sq_dist.shape)
        if len(sq_dist) == 0:
            break
        else:
            indices = np.argmin(sq_dist, axis=0)
            print(indices[0])
            print(indices.shape)
            nearest_neighbors = [np.unravel_index(i, sq_dist.shape) for i in indices]
            print(nearest_neighbors[2])
            # assert len(nearest_neighbors) == len(keep) # one nearest neighbor for each center?
            for i in range(len(keep)):
                center = keep[i]
                neighbor = nearest_neighbors[i]
                # if h_strengths[keep] < 0.9
            keep.append(nearest_neighbors)
            for n in nearest_neighbors:
                candidates.remove(n)
            # candidates.remove(nearest_neighbors)

        r += constants.MIN_RADIUS

    #     assert len(keep) == constants.NUM_KEEP
    # return h_strengths, coords[:constants.NUM_KEEP]
    return keep


def anms_2(strength, coords):
    selected_indices = []
    candidates = [(coord[0], coord[1]) for coord in coords]
    dists = utils.dist2(coords, coords)

    max_global = float("-inf")
    max_global_index = None
    for index in range(len(coords)):
        x = coords[index][0]
        y = coords[index][1]
        if strength[y, x] > max_global:
            max_global_index = index
            max_global = strength[y, x]

    selected_indices = [max_global_index]

    # add nearest neighbors repeatedly
    for r in reversed(range(constants.MIN_RADIUS, constants.MAX_RADIUS)):
        for candidate_index in range(len(candidates)):
            isGood = True
            for good_index in selected_indices:
                if dists[candidate_index, good_index] < r * r:
                    isGood = False
                    break
            if isGood:
                selected_indices.append(candidate_index)
                if len(selected_indices) >= constants.NUM_KEEP:
                    break
        if len(selected_indices) >= constants.NUM_KEEP:
            break

    assert len(selected_indices) == constants.NUM_KEEP

    selected_coords = np.array([coords[i] for i in selected_indices])
    utils.assert_coords(selected_coords)
    return selected_coords
