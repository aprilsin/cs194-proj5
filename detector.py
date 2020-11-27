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
    return h, np.flip(coords, axis=-1)  # to get (x, y)


def anms_ignore(h_strengths, coords, eps=0.9) -> list:
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

    assert (
        len(selected_indices) == constants.NUM_KEEP
    ), f"expected {constants.NUM_KEEP} points but got {len(selected_indices)}"

    selected_coords = np.array([coords[i] for i in selected_indices])
    utils.assert_coords(selected_coords)
    return selected_coords


def anms(strength, detected_coords, robust_factor=0.9):
    """
    Everything in this function works with indices to detected_coords.
    Returns top NUM_KEEP points from detected_coords.
    """

    # sort by strength
    print(detected_coords[0], type(detected_coords[0]))
    detected_coords = sorted(detected_coords, key=lambda i: strength[i[1], i[0]])
    print(detected_coords[0], type(detected_coords[0]))

    selected = []
    all_candidates = [*range(len(detected_coords))]
    r = constants.MAX_RADIUS  # initialize suppression radius to infinity

    # add global maximum
    selected = [0]
    all_candidates.remove(0)

    while (
        len(selected) < constants.NUM_KEEP and r > constants.MIN_RADIUS
    ):  # TODO can I do this???
        for selected_ind in selected:
            coord = detected_coords[selected_ind]
            candidates = [detected_coords[i] for i in all_candidates]
            dists = np.sqrt(utils.dist2(coord, candidates)).T

            # keep if candidate is outside of supression index
            candidates = [i for i in range(len(dists)) if dists[i] > r]
            candidates = [1, 2, 3]
            for i in candidates:
                candidate_coord = detected_coords[i]
                if (
                    strength[coord[1], coord[0]]
                    < robust_factor * strength[candidate_coord[1], candidate_coord[0]]
                ):
                    if len(selected) < constants.NUM_KEEP:
                        selected.append(i)
                    else:
                        break  # TODO need to break out of two loops
        r -= 1

    selected_coords = np.array([detected_coords[i] for i in selected])
    utils.assert_coords(selected_coords)
    return
