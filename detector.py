import itertools
from dataclasses import dataclass
from functools import total_ordering
from queue import PriorityQueue
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
from skimage.feature import corner_harris, corner_peaks, peak_local_max
import sys  # TODO remove this
import filters
import homography
import utils
from constants import *


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
    # im = filters.gauss_blur(im)

    # find harris corners
    h = corner_harris(im, method="eps", sigma=1)
    coords = peak_local_max(
        h, min_distance=1, indices=True, threshold_rel=MIN_HARRIS_STRENGTH
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
    return h, coords


def anms(h_strengths, coords, eps=0.9) -> list:

    # initialize
    keep = []
    candidates = [(coord[0], coord[1]) for coord in coords]
    r = MIN_RADIUS  # suppression radius

    # get global maximum
    strongest_corner = None
    strongest_strength = 0
    for corner in candidates:
        if h_strengths[corner] > strongest_strength:
            strongest_corner = corner
            strongest_strength = h_strengths[corner]

    keep.append(strongest_corner)
    candidates.remove(strongest_corner)

    while len(keep) < NUM_KEEP and len(candidates) > 0 and r < MAX_RADIUS:
        # compute ssd for all kept centers / coords
        sq_dist = utils.dist2(keep, candidates)
        # outlier rejection
        mask = np.where(sq_dist <= r)
        sq_dist = sq_dist[mask]
        if len(sq_dist) == 0:
            break
        else:
            nearest_neighbors = np.unravel_index(
                np.argmin(sq_dist, axis=0), sq_dist.shape
            )
            # assert len(nearest_neighbors) == len(keep) # one nearest neighbor for each center?
            print(nearest_neighbors)
            keep.append(nearest_neighbors)
            candidates.remove(nearest_neighbors)
        # for center in keep:
        #     sq_dist = utils.dist2(center, candidates)

        #     # outlier rejection
        #     mask = np.where(sq_dist <= r)
        #     sq_dist = sq_dist[mask]
        #     if len(sq_dist) == 0:
        #         break
        #     else:
        #         nearest_neighbor = np.unravel_index(np.argmin(sq_dist), sq_dist.shape)
        #         keep.append(nearest_neighbor)
        #         candidates.remove(nearest_neighbor)
        r += MIN_RADIUS

    sys.exit()  # TODO remove this
    #     assert len(keep) == NUM_KEEP
    # return h_strengths, coords[:NUM_KEEP]
    return keep


# def get_corners(img):
#     h_strengths, coords = get_harris(img)
#     corners = anms(h_strengths, coords)
#     return corners
