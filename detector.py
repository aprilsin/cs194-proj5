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
    print(type(coords), coords[0])
    # discard points on edge
    edge = edge_discard  # pixels
    mask = (
        (coords[:, 0] > edge)
        & (coords[:, 0] < im.shape[0] - edge)
        & (coords[:, 1] > edge)
        & (coords[:, 1] < im.shape[1] - edge)
    )
    coords = coords[mask]
    print(np.array([0, 0]) in coords)
    # return h, coords
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

    r = MIN_RADIUS  # suppression radius
    while len(keep) < NUM_KEEP and len(candidates) > 0 and r < MAX_RADIUS:
        # compute ssd for all kept centers / coords
        sq_dist = utils.dist2(keep, candidates)
        print(sq_dist)
        # outlier rejection
        mask = np.where(sq_dist <= 500)
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

        r += MIN_RADIUS

    #     assert len(keep) == NUM_KEEP
    # return h_strengths, coords[:NUM_KEEP]
    return keep


def anms_2(strength, coords):
    selected_indices = []
    candidates = [(coord[0], coord[1]) for coord in coords]
    dists = utils.dist2(coords, coords)

    # find global maximum
    # strongest_corner = None
    # strongest_strength = float("-inf")
    # # for corner in candidates:
    # #     if strength[corner] > strongest_strength:
    # #         strongest_corner = corner
    # #         strongest_strength = strength[corner]
    # # selected.append(strongest_corner)

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
    for r in reversed(range(MIN_RADIUS, MAX_RADIUS)):
        for candidate_index in range(len(candidates)):
            isGood = True
            for good_index in selected_indices:
                if dists[candidate_index, good_index] < r * r:
                    isGood = False
                    break
            if isGood:
                selected_indices.append(candidate_index)
                # print(
                #     "Found "
                #     + str(len(selected))
                #     + " out of "
                #     + str(NUM_KEEP)
                #     + " points expected."
                # )
                if len(selected_indices) >= NUM_KEEP:
                    break
        if len(selected_indices) >= NUM_KEEP:
            break

    # figure, axis = plt.subplots(ncols=3)
    # axis[0].imshow(img, vmin=0, vmax=1)
    # axis[1].imshow(harris_img, vmin=0, vmax=1)
    # axis[2].imshow(img, vmin=0, vmax=1)

    # for x, y in points:
    #     marker = plt.Circle((x, y), 2, color="r")
    #     axis[2].add_artist(marker)

    # plt.show()

    # selected = [np.unravel_index(i, strength.shape) for i in selected]
    selected_coords = np.array([coords[i] for i in selected_indices])
    assert selected_coords.shape == (NUM_KEEP, 2)
    return selected_coords


# def get_corners(img):
#     h_strengths, coords = get_harris(img)
#     corners = anms(h_strengths, coords)
#     return corners
