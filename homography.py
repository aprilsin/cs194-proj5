import itertools

import numpy as np
import skimage as sk
import matplotlib.pyplot as plt

# import skimage.io as io
from scipy import interpolate
import cv2
from my_types import assert_points


def homo_matrix(im1_pts: np.ndarray, im2_pts: np.ndarray):
    """Returns a homographic transformation matrix from ptsA to ptsB"""
    assert len(im1_pts) == len(im2_pts)

    num_pts = len(im1_pts)

    A = []
    B = []
    for (x1, y1), (x2, y2) in zip(im1_pts, im2_pts):
        row1 = [x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2]
        row2 = [0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2]
        # row1 = [-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2]
        # row2 = [0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2]
        A.append(row1)
        A.append(row2)
        B.extend((x2, y2))

    A = np.array(A)
    B = np.array(B)

    params, *_ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c, d, e, f, g, h = params
    H = np.array([[a, b, c], [d, e, f], [g, h, 1]])
    return H


def bounding_box(img, H):
    """ Return the corners of the projected image in H, W domain """
    h, w, c = img.shape
    corners_hw = [[0, 0], [0, w - 1], [h - 1, w - 1], [h - 1, 0]]
    corners_xy = [[c, r] for r, c in corners_hw]  # x, y = c, r

    corners = [[x, y, 1] for x, y in corners_xy]
    corners = np.array(corners).T  # so that each column is [x, y, 1]

    bounds = H @ corners

    bounds /= bounds[2]  # fix w, scaling due to transformation
    bounds = bounds.T[:, :2]  # reshape into the form [[x1, y1], [x2, y2]]

    # make sure indices are integers
    bounds = np.ceil(bounds).astype(np.int64)

    # shift indices to zero-indexed
    min_x = bounds[:, 0].min()
    min_y = bounds[:, 1].min()

    bounds[:, 0] += -min_x
    bounds[:, 1] += -min_y

    bounds = np.flip(bounds, axis=1)  # flip x, y to r, c
    row_shift = -min_y
    col_shift = -min_x
    return bounds, row_shift, col_shift


def warp_mask(img, h_matrix) -> np.ndarray:
    assert h_matrix.shape == (3, 3)

    h, w, c = img.shape
    H, W = range(h), range(w)

    # initialize warped img matrix
    box, row_shift, col_shift = bounding_box(img, h_matrix)
    bound_rows = box[:, 0]
    bound_cols = box[:, 1]
    mask_h, mask_w = bound_rows.max() + 1, bound_cols.max() + 1
    # + 1 since the bounding box needs to be a valid index
    mask = np.zeros((mask_h, mask_w, c))

    # compute target coordinates
    target_rr, target_cc = sk.draw.polygon(
        bound_rows, bound_cols, shape=(mask_h, mask_w)
    )
    mask[target_rr, target_cc] = 1
    mask = np.clip(mask, 0.0, 1.0)
    return mask


def empty_warp(img, h_matrix):
    h, w, c = img.shape
    box, _, _ = bounding_box(img, h_matrix)

    warp_h, warp_w = box[:, 0].max() + 1, box[:, 1].max() + 1
    # + 1 so the bounding box will be a valid index

    return np.zeros((warp_h, warp_w, c))


def forward_warp(img, h_matrix) -> np.ndarray:
    assert h_matrix.shape == (3, 3)

    h, w, ch = img.shape
    H, W = range(h), range(w)

    # initialize warped img matrix
    warped = empty_warp(img, h_matrix)

    # Compute Source Coordinates
    print("=====src=====")
    # coordinates = np.meshgrid(H,W) # for each pixel
    coordinates = np.array(list(itertools.product(H, W)))
    src_rr, src_cc = coordinates[:, 0], coordinates[:, 1]

    print(img.shape)
    print(src_rr.min(), src_cc.min())
    print(src_rr.max(), src_cc.max())

    # Compute Target Coordinates
    print("====target====")
    pts_3D = [[c, r, 1] for r, c in coordinates]  # x, y = c, r
    pts_3D = np.array(pts_3D).T  # so that each column is [x, y, 1]

    target_pts = h_matrix @ pts_3D
    target_pts /= target_pts[2]  # fix w, scaling due to transformation

    target_cc, target_rr = target_pts.T[:, 0], target_pts.T[:, 1]  # c, r = x, y

    # make sure indices are integers
    target_rr = np.int32(np.round(target_rr))
    target_cc = np.int32(np.round(target_cc))
    # shift indices to zero-indexed
    target_rr += -target_rr.min()
    target_cc += -target_cc.min()

    print(warped.shape)
    print(target_rr.min(), target_cc.min())
    print(target_rr.max(), target_cc.max())

    # make sure indicies are in-bounds
    target_rr = np.clip(target_rr, 0, warped.shape[0] - 1)
    target_cc = np.clip(target_cc, 0, warped.shape[1] - 1)

    # Do interpolation
    print("=====interpolate=====")
    interp_funcs = [
        interpolate.RectBivariateSpline(H, W, img[:, :, c]) for c in range(ch)
    ]
    for i, f in enumerate(interp_funcs):
        # x = rows, y = cols for forward warping
        warped[target_rr, target_cc, i] = f.ev(xi=src_rr, yi=src_cc)

    warped = np.clip(warped, 0.0, 1.0)
    return warped


def fill_holes(warped, src_img, h_matrix):
    """ fill holes formed due to forward warping """
    bound_mask = warp_mask(src_img, h_matrix)
    mask = np.ma.make_mask((bound_mask == 1) & (warped == 0)).astype(np.uint8)

    warped = warped.astype(np.float32)
    filled = np.zeros_like(warped)

    for ch in range(3):
        w_ch = warped[:, :, ch].astype(np.float32)
        m_ch = mask[:, :, ch]
        filled_ch = cv2.inpaint(w_ch, m_ch, 3, cv2.INPAINT_NS)
        filled[:, :, ch] = filled_ch

    return filled


def inverse_warp(img, h_matrix) -> np.ndarray:
    assert h_matrix.shape == (3, 3)

    h, w, c = img.shape
    H, W = range(h), range(w)

    # initialize warped img matrix
    box, row_shift, col_shift = bounding_box(img, h_matrix)
    bound_rows = box[:, 0]
    bound_cols = box[:, 1]
    warp_h, warp_w = bound_rows.max() + 1, bound_cols.max() + 1
    # + 1 since the bounding box needs to be a valid index
    warped = np.zeros((warp_h, warp_w, c))

    # compute target coordinates
    print("====target====")
    target_rr, target_cc = sk.draw.polygon(
        bound_rows, bound_cols, shape=(warp_h, warp_w)
    )

    print(warped.shape)
    print(target_rr.min(), target_cc.min())
    print(target_rr.max(), target_cc.max())

    # compute source coordinates
    print("=====src=====")

    num_pts = len(target_rr)
    target_pts = np.vstack((target_cc, target_rr, np.ones((1, num_pts))))

    src_pts = np.linalg.inv(h_matrix) @ target_pts
    src_rr, src_cc = (
        src_pts.T[:, 0],
        src_pts.T[:, 1],
    )  # in correct order with r, c since inversed

    # make sure indices are integers
    src_rr = np.int32(np.round(src_rr))
    src_cc = np.int32(np.round(src_cc))
    # shift indices to zero-indexed
    src_rr += -src_rr.min()
    src_cc += -src_cc.min()

    print(img.shape)
    print(src_rr.min(), src_cc.min())
    print(src_rr.max(), src_cc.max())

    # interpolate
    print("=====interpolate=====")
    interp_funcs = [
        interpolate.RectBivariateSpline(H, W, img[:, :, c]) for c in range(3)
    ]
    for i, f in enumerate(interp_funcs):
        warped[target_rr, target_cc, i] = f.ev(xi=src_cc, yi=src_rr)

    return warped
