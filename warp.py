# morphing sequence

import copy
import math
import time
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as io
from matplotlib import animation
from scipy import interpolate
from scipy.spatial import Delaunay
from skimage import transform
from skimage.util import img_as_float, img_as_ubyte

import utils
from my_types import *

#######################
#    DEFINE SHAPES    #
#######################


def weighted_avg(im1_pts: np.ndarray, im2_pts: np.ndarray, alpha) -> np.ndarray:
    """
    Compute the (weighted) average points of correspondence
    """
    assert len(im1_pts) == len(im2_pts), (len(im1_pts), len(im2_pts))
    return alpha * im1_pts + (1 - alpha) * im2_pts


def delaunay(points):
    return Delaunay(points)


def points_from_delaunay(points, triangulation):
    return points[triangulation.simplices]


def plot_tri_mesh(img: np.ndarray, points: np.ndarray, triangulation) -> None:
    """
    Displays the triangular mesh of an image
    """
    plt.imshow(img)
    plt.triplot(points[:, 0], points[:, 1], triangulation.simplices)
    plt.plot(points[:, 0], points[:, 1], "o")
    plt.show()


###############################
#   WARP AND CROSS DISSOLVE   #
###############################


def inverse_affine_matrix(start: Triangle, target: Triangle) -> np.ndarray:
    assert_is_triangle(start)
    assert_is_triangle(target)

    A = np.vstack((start[:, 0], start[:, 1], [1, 1, 1]))
    try:
        inv = np.linalg.inv(A)
    except:
        return
    B = np.vstack((target[:, 0], target[:, 1], [1, 1, 1]))
    # B = T * A
    # T = B * A^-1
    T = B @ inv
    return np.linalg.inv(T)


def inverse_pixels(img, img_triangle_vertices, target_triangle_vertices):
    """ Returns the coordinates of pixels from original image. """
    assert_img_type(img)
    assert_is_triangle(img_triangle_vertices)
    assert_is_triangle(target_triangle_vertices)

    inv_affine_mat = inverse_affine_matrix(
        img_triangle_vertices, target_triangle_vertices
    )

    x, y, _ = img.shape
    target_rr, target_cc = sk.draw.polygon(
        target_triangle_vertices.T[0],
        target_triangle_vertices.T[1],
        shape=(y, x),
    )
    # Transform points to the source image domain
    target_points = np.vstack(
        (target_rr, target_cc, np.ones(len(target_cc)))
    )  # append 1 to all rows?
    src_points = inv_affine_mat @ target_points

    return src_points


def warp_img(
    img: np.ndarray,
    img_pts: np.ndarray,
    target_pts: np.ndarray,
    triangulation: Delaunay,
) -> np.ndarray:
#     assert_img_type(img)
#     assert_points(img_pts)
#     assert_points(target_pts)

    h, w, c = img.shape
    warped = np.zeros_like(img)

    # Interpolation functions
    f_red = interpolate.RectBivariateSpline(
        range(img.shape[0]), range(img.shape[1]), img[:, :, 0]
    )
    f_green = interpolate.RectBivariateSpline(
        range(img.shape[0]), range(img.shape[1]), img[:, :, 1]
    )
    f_blue = interpolate.RectBivariateSpline(
        range(img.shape[0]), range(img.shape[1]), img[:, :, 2]
    )

    for simplex in triangulation.simplices:

        target_vertices = target_pts[simplex]
        img_vertices = img_pts[simplex]

        # Transform points to the source image domain with inverse warping
        h, w, _ = img.shape
        target_rr, target_cc = sk.draw.polygon(
            target_vertices.T[0],
            target_vertices.T[1],
            shape=(w, h),
        )
        src_points = inverse_pixels(img, img_vertices, target_vertices)

        # Interpolate
        warped[target_cc, target_rr, 0] = f_red.ev(src_points[1], src_points[0])
        warped[target_cc, target_rr, 1] = f_green.ev(src_points[1], src_points[0])
        warped[target_cc, target_rr, 2] = f_blue.ev(src_points[1], src_points[0])

    warped = np.clip(warped, 0.0, 1.0)
    assert_img_type(warped)
    return warped


def cross_dissolve(warped_im1, warped_im2, alpha):
    return weighted_avg(warped_im1, warped_im2, alpha=alpha)


def compute_middle_object(
    im1: ToImgArray,
    im2: ToImgArray,
    im1_pts: ToPoints,
    im2_pts: ToPoints,
    alpha,
):
    im1 = to_img_arr(im1)
    im2 = to_img_arr(im2)
    im1_pts = to_points(im1_pts)
    im2_pts = to_points(im2_pts)

    mid_pts = weighted_avg(im1_pts, im2_pts, alpha=alpha)
    triangulation = delaunay(mid_pts)

    im1_warped = warp_img(im1, im1_pts, mid_pts, triangulation)
    im2_warped = warp_img(im2, im2_pts, mid_pts, triangulation)

    middle_img = cross_dissolve(im1_warped, im2_warped, alpha=alpha)
    # middle_img = transform.rotate(middle_img, -90)
    return middle_img, mid_pts, triangulation