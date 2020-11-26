import itertools
from dataclasses import dataclass

import numpy as np
import skimage.transform
from skimage.feature import corner_harris, corner_peaks

import filters
import homography
import utils

# from constants import NUM_KEEP


@dataclass
class Feature:
    coord: np.ndarray  # shape = (2,  )
    patch: np.ndarray  # shape = (8, 8)


def get_corners(im, edge_discard=20):
    """
    Finds all harris corners in the image. Harris corners near the edge are discarded and the coordinates of the remaining corners are returned.

    Input:
    Takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels).

    Output:
    - a 2d array (h_strengths) of the same shape as the original image (im) containing the Harris corner strength of every pixel.
    - coords is n x 2 (xs, ys).
    """

    assert edge_discard >= 20

    # find harris corners
    h_strengths = corner_harris(im, method="eps", sigma=1)
    coords = corner_peaks(h_strengths, min_distance=8, indices=True, threshold_rel=0)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (
        (coords[:, 0] > edge)
        & (coords[:, 0] < im.shape[0] - edge)
        & (coords[:, 1] > edge)
        & (coords[:, 1] < im.shape[1] - edge)
    )

    # return h, np.flip(coords[mask], axis=1) # [x, y] = [c, r]
    return coords[mask], h_strengths


def detect_corners(img):
    # num_levels = 5
    # g_stack = filters.gaussian_stack(img)
    # for level in g_stack:
    assert img.ndim == 3, img.shape
    blurred = filters.gauss_blur(img)
    harris_strengths, coords = get_corners(utils.to_gray(blurred))
    return harris_strengths, coords


def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points in polar coordinates.

    Input:
    - Takes two matrices of vectors and calculates the squared Euclidean distance between them.
    - Both matrices must be of the same column dimension.

    Output:
    - If X has M rows and N columns, and C has L rows and N columns, then the result has M rows and L columns.  The I, Jth entry is the squared distance from the Ith row of X to the Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """

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


def anms(detected_corners, corners_strengths):
    mask = np.full(shape=corners_strengths.shape, fill_value=-float("inf"))
    NUM_GLOBAL = 10
    indices = (-corners_strengths).argsort()[:NUM_GLOBAL]
    mask[detected_corners[:, 0], detected_corners[:, 1]] = corners_strengths[
        detected_corners[:, 0], detected_corners[:, 1]
    ]

    keep = set()
    r = 0  # initialize suppression radius
    NUM_KEEP = 100  # want to keep the best 500 corners
    while len(keep) < NUM_KEEP:
        strongest_corner = np.argmax(corners_strengths)  # TODO
        keep.add(strongest_corner)
    return np.array(list(keep))


def vectorize(patch):
    assert patch.shape == (8, 8), patch.shape
    return np.reshape(patch, (1, 64))


def normalize(mat):
    return (mat - np.mean(mat)) / np.std(mat)


def refine_matches():
    pass


def get_patches(img, corners) -> np.ndarray:
    """make all detected corner an 8x8 grid"""
    img = filters.gauss_blur(img)
    img = utils.to_gray(img)  # TODO make it work with colored patches

    patches = []
    for (r, c) in corners:
        patch = img[r - 20 : r + 20, c - 20 : c + 20]  # get a 40x40 patch

        # patch = skimage.transform.resize(patch, (8, 8))  # downsample
        patch = patch[::5, ::5, :]  # downsample by taking every 5th pixel

        patch = normalize(patch)  # normalize

        patches.append(patch)

    return np.array(patches)


def dist_patches(patch1, patch2):
    """
    patch1 and patch2 are 8x8 grids.
    """
    assert patch1.shape == patch2.shape == (8, 8), (patch1.shape, patch2.shape)
    patch1 = np.reshape(patch1, (1, 64))
    patch2 = np.reshape(patch2, (1, 64))
    return np.sum((patch1 - patch2) ** 2)  # sum squared distance


def match_features(features1, features2):
    combos = list(itertools.product(features1, features2))
    
    return matched1, matcheds2


def ransac(matched_corners1, matched_corners2, epsilon):
    assert len(matched_corners1) == len(matched_corners2), (
        len(matched_corners1),
        len(matched_corners2),
    )

    num_input_matches = len(matched_corners1)
    max_inliers = 0
    best_inliers1, best_inliners2 = [], []

    # select 4 points at random (4 points are needed to compute homography)
    for indicies in itertools.combinations(range(num_input_matches), 4):
        # compute homography
        corners1, corners2 = matched_corners1[indices], matched_corners2[indices]
        h_matrix = homography.homo_matrix(corners1, corners2)

        # compute inliers
        predicted2 = homography.warp_pts(corners2, h_matrix)
        dist = dist2(corners2, predicted2)

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
