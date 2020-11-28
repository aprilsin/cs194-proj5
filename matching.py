import itertools
import random
import sys  # TODO remove this
from collections import Counter, defaultdict

import numpy as np
import skimage.transform
from scipy.spatial.distance import cdist, sqeuclidean
from tqdm import trange

import constants
import filters
import homography
import utils
from constants import MATCHING_THRESHOLD, RANSAC_THRESHOLD


def match_features(coords1, patches1, coords2, patches2, threshold=MATCHING_THRESHOLD):
    print(f"Matching {len(coords1)} features with {len(coords2)} features.")
    assert len(coords1) == len(patches1), f"{len(coords1)}, {len(patches1)}"
    assert len(coords2) == len(patches2), f"{len(coords2)}, {len(patches2)}"
    utils.assert_coords(coords1)
    utils.assert_coords(coords2)

    ssd = cdist(patches1, patches2, metric="sqeuclidean")
    # XXX this is necessary to ensure that the mapping is a bijection so it can
    # be used for homography later.
    used = {}
    matched_indices = set()

    for i in trange(len(patches1)):
        j_first, j_second, *_ = np.argsort(ssd[i]).ravel()
        ratio = ssd[i, j_first] / ssd[i, j_second]
        if ratio < threshold:
            if j_first in used:
                i_, j_second_, ratio_ = used[j_first]
                if ratio < ratio_:
                    used[j_first] = [i, j_second, ratio]
                    matched_indices.remove((i_, j_first))
                    matched_indices.add((i, j_first))
            else:
                used[j_first] = [i, j_second, ratio]
                matched_indices.add((i, j_first))

    matched1 = np.array([coords1[i] for i, _ in matched_indices])
    matched2 = np.array([coords2[j] for _, j in matched_indices])

    if constants.DEBUG:
        print(matched1[0], type(matched1[0]))
    assert len(matched1) == len(matched2)
    utils.assert_coords(matched1)
    utils.assert_coords(matched2)
    return matched1, matched2


def ransac(corners1, corners2, epsilon=3):
    assert len(corners1) == len(corners2), (
        len(corners1),
        len(corners2),
    )

    corners1 = np.array(corners1)
    corners2 = np.array(corners2)
    print(f"{corners1.shape=}")
    print(f"{corners2.shape=}")
    assert corners1.ndim == 2 and corners2.ndim == 2

    best_num_inliers = 0
    best_inliers1, best_inliers2 = [], []

    # select NUM_SAMPLE_POINTS points at random to compute homography
    for _ in range(10_000):
        rand_idxs = np.random.choice(len(corners1), replace=False, size=4)
        chosen1, chosen2 = corners1[rand_idxs], corners2[rand_idxs]
        # compute homography
        h_matrix = homography.homo_matrix(chosen1, chosen2)

        # compute inliers and count number of coordinates that are good matches
        predicted2 = homography.warp_pts(corners1, h_matrix)
        # compare predicted with ground truth
        dist = np.sqrt(utils.ssd_points(corners2, predicted2))

        if constants.DEBUG:
            print(min(dist), max(dist))

        matches = dist < epsilon

        num_matches = matches.sum()
        if constants.DEBUG:
            print(num_matches)

        # save inliers if they are the largest set so far
        if num_matches > best_num_inliers:
            best_inliers1 = corners1[matches]
            best_inliers2 = corners2[matches]
            best_num_inliers = num_matches

    assert len(best_inliers1) == len(best_inliers2)
    best_inliers1 = np.array(best_inliers1)
    best_inliers2 = np.array(best_inliers2)

    utils.assert_coords(best_inliers1)
    utils.assert_coords(best_inliers2)
    return best_inliers1, best_inliers2
