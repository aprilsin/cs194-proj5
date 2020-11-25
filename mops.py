import itertools

import numpy as np
from skimage.feature import corner_harris, corner_peaks

import filters
import homography
import utils


def get_corners(im, edge_discard=20):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is n x 2 (xs, ys).
    """

    assert edge_discard >= 20

    # find harris corners
    h = corner_harris(im, method="eps", sigma=1)
    coords = corner_peaks(h, min_distance=2, indices=True, threshold_rel=0)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (
        (coords[:, 0] > edge)
        & (coords[:, 0] < im.shape[0] - edge)
        & (coords[:, 1] > edge)
        & (coords[:, 1] < im.shape[1] - edge)
    )

    # return h, np.flip(coords[mask], axis=1) # [x, y] = [c, r]
    return h, coords


def detect_corners(img):
    # num_levels = 5
    # g_stack = filters.gaussian_stack(img)
    # for level in g_stack:
    assert img.ndim == 3, img.shape
    blurred = filters.gauss_blur(img)
    harris_strengths, coords = harris.get_corners(utils.to_gray(blurred))
    return harris_strengths, coords


def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """

    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, "Data dimension does not match dimension of centers"
    sq_dist = ( # (x1, y1)^2 - (x2, y2)^2
        (np.ones((ncenters, 1)) * np.sum((x ** 2).T, axis=0)).T
        + np.ones((ndata, 1)) * np.sum((c ** 2).T, axis=0)
        - 2 * np.inner(x, c)
    )
    return sq_dist


def refine_matches():
    pass


def ransac(matched_corners1, matched_corners2, epsilon):
    assert len(matched_corners1) == len(matched_corners2), (
        len(matched_corners1),
        len(matched_corners2),
    )

    # select 4 points at random (4 points are needed to compute homography)
    num_input_matches = len(matched_corners1)
    max_inliers = 0
    best_inliers1, best_inliners2 = [], []
    for indicies in itertools.combinations(range(num_input_matches), 4):
        # compute homography
        corners1, corners2 = matched_corners1[indices], matched_corners2[indices]
        h_matrix = homography.homo_matrix(corners1, corners2)
        # compute inliers
        predicted2 = homography.warp_pts(corners2, h_matrix)
        dist = dist2(corners2, predicted2)
        # count number of coordinates that are good matches
        num_matches = np.cound_non_zero(dist < epsilon)
        # save inliners if they are the largest set so far
        if num_matches > max_inliers:
            best_inliers1, best_inliers2 = np.where(dist)

    pass


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
        patch = img[r - 20 : r + 20, c - 20 : c + 20]
        # downsample
        patch = skimage.transform.resize(patch, (8, 8))
        # normalize
        patch -= mean(patch)
        patch /= np.std(patch)
        patches.append(patch)
    return patches
