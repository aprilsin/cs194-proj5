import itertools

import numpy as np
import skimage.transform

import filters
import homography
import utils
from constants import *


def match_features(features1, features2):
    combos = list(itertools.product(features1, features2))
    ssd = [utils.dist_patches(f1.patch, f2.patch) for f1, f2 in combos]
    
    matched1, matched2 = features1, features2
    return matched1, matched2


def ransac(matched_corners1, matched_corners2, epsilon):
    assert len(matched_corners1) == len(matched_corners2), (
        len(matched_corners1),
        len(matched_corners2),
    )

    num_input_matches = len(matched_corners1)
    max_inliers = 0
    best_inliers1, best_inliners2 = [], []

    # select 4 points at random (4 points are needed to compute homography)
    for indices in itertools.combinations(range(num_input_matches), 4):
        # compute homography
        corners1, corners2 = matched_corners1[indices], matched_corners2[indices]
        h_matrix = homography.homo_matrix(corners1, corners2)

        # compute inliers
        predicted2 = homography.warp_pts(corners2, h_matrix)

        import detector

        dist = detector.dist2(corners2, predicted2)  # TODO fix ssd

        # count number of coordinates that are good matches
        matches = np.where(dist < epsilon)
        num_matches = np.cound_non_zero(matches)

        # save inliners if they are the largest set so far
        if num_matches > max_inliers:
            best_inliers1, best_inliers2 = (
                matched_corners1[matches[:, 0]],
                matched_corners2[matches[:, 1]],
            )
            max_inliers = num_matches

    return best_inliers1, best_inliers2