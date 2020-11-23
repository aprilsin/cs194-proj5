import filters, harris
import itertools
import numpy as np
import utils
import numpy as np
from skimage.feature import corner_harris, corner_peaks


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
    coords = corner_peaks(h, min_distance=2, indices=True)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (
        (coords[:, 0] > edge)
        & (coords[:, 0] < im.shape[0] - edge)
        & (coords[:, 1] > edge)
        & (coords[:, 1] < im.shape[1] - edge)
    )

    return h, np.flip(coords[mask], axis=1)


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

    return (
        (np.ones((ncenters, 1)) * np.sum((x ** 2).T, axis=0)).T
        + np.ones((ndata, 1)) * np.sum((c ** 2).T, axis=0)
        - 2 * np.inner(x, c)
    )


def detect_corners(img):
    # num_levels = 5
    # g_stack = filters.gaussian_stack(img)
    # for level in g_stack:
    assert img.ndim == 3, img.shape
    blurred = filters.gauss_blur(img)
    harris_stength, coords = harris.get_corners(utils.to_gray(blurred))
    return coords


def match_features(im1_grids, im2_grids):
    matched = []
    for i, j in itertools.product(len(im1_grids), len(im2_grids)):
        if np.equals(im1_grids[i], im2_grids[i]):
            matched.append([i, j])
    return matched


def standardize():
    """make all detected corner an 8x8 grid"""
    pass
