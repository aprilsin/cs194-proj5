import itertools
from dataclasses import dataclass
from functools import total_ordering
from queue import PriorityQueue

import numpy as np
import skimage.transform
from skimage.feature import corner_harris, corner_peaks, peak_local_max

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
    im = filters.gauss_blur(im)

    # find harris corners
    h = corner_harris(im, method='eps', sigma=1)
    coords = peak_local_max(h, min_distance=1, indices=True, threshold_rel=MIN_HARRIS_STRENGTH)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
    (coords[:, 0] < im.shape[0] - edge) & \
    (coords[:, 1] > edge) & \
    (coords[:, 1] < im.shape[1] - edge)
    
    coords = coords[mask]
    return h, coords


def dist2(x: list, c: list) -> np.ndarray:
    """
    dist2  Calculates squared distance between two sets of points in polar coordinates.

    Input:
    - Takes two matrices of vectors and calculates the squared Euclidean distance between them.
    - Both matrices must be of the same column dimension.

    Output:
    - If X has M rows and N columns, and C has L rows and N columns, then the result has M rows and L columns.  The I, Jth entry is the squared distance from the Ith row of X to the Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """
    x, c = np.array([a.coord for a in x]), np.array([a.coord for a in c])
    print(x)
    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, "Data dimension does not match dimension of centers"

    # dist^2 = r^2 + s^2 - 2*rs*cos(theta-phi)
    # dist^2 = r^2 + s^2 - 2*inner-product
    r_sq = np.ones((ncenters, 1)) * np.sum((x ** 2).T, axis=0)
    s_sq = np.ones((ndata, 1)) * np.sum((c ** 2).T, axis=0)

    sq_dist = (
        (np.ones((ncenters, 1)) * np.sum((x ** 2).T, axis=0)).T
        + np.ones((ndata, 1)) * np.sum((c ** 2).T, axis=0)
        - 2 * np.inner(x, c)
    )
    return sq_dist


def anms(h_strengths, coords, eps=0.9) -> list:

    # initialize
    keep = []
    r = MIN_RADIUS  # suppression radius

    # get global maximum
    x = np.argmax(h_strengths)
    index = np.unravel_index(x, h_strengths.shape)
    print(x, index, h_strengths[index])
#     keep.append(strongest_corner)
#     detected_corners.remove(strongest_corner)

#     while len(keep) < NUM_KEEP and len(candidates) > 0 and r < MAX_RADIUS:
#         for center in keep:
#             sq_dist = dist2(centers, candidates)
#             mask = np.where(sq_dist <= r)
#             indices = np.argmax(sq_dist[mask])  # TODO check axis
#             best_candidate = candidates[indices]
#             keep.append(best_candidate)
#             for c in best_candidate:
#                 print(c)
#                 candidates.remove(best_candidate[c])

#     assert len(keep) == NUM_KEEP
    return h_strengths, coords


def get_corners(img):
    h_strengths, coords = get_harris(img)
    corners = anms(h_strengths, coords)
    return corners
