import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk

# import skimage.io as io
from scipy import interpolate

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


def warp_pts(pts, h_matrix, pts_shift):
    pts_3D = [[x, y, 1] for x, y in pts]  # x, y = c, r
    pts_3D = np.array(pts_3D).T  # so that each column is [x, y, 1]
    target_pts = h_matrix @ pts_3D
    target_pts /= target_pts[2]  # fix w, scaling due to transformation
    return np.add(target_pts.T[:, :2], pts_shift)


def forward_warp(img, h_matrix, fill=True) -> np.ndarray:
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
    row_shift = -target_rr.min()
    col_shift = -target_cc.min()
    target_rr += row_shift
    target_cc += col_shift

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

    # fill gaps caused by forward warping
    if fill:
        warped = fill_holes(warped, img, h_matrix)

    warped = np.clip(warped, 0.0, 1.0)
    return warped, [col_shift, row_shift]  # x, y = c, r


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


def bounding_box(img, H):
    """ Return the corners of the projected image in H, W domain """
    h, w, c = img.shape
    corners_hw = [
        [0, 0],
        [0, w - 1],
        [h - 1, w - 1],
        [h - 1, 0],
    ]  # in this order for polygon
    corners_xy = [[c, r] for r, c in corners_hw]  # x, y = c, r

    corners = [[x, y, 1] for x, y in corners_xy]
    corners = np.array(corners).T  # so that each column is [x, y, 1]

    bounds = H @ corners

    bounds /= bounds[2]  # fix w, scaling due to transformation
    bounds = bounds.T[:, :2]  # reshape into the form [[x1, y1], [x2, y2]]

    # make sure indices are integers
    bounds = np.round(bounds).astype(np.int64)

    # shift indices to zero-indexed
    x_shift = -bounds[:, 0].min()
    y_shift = -bounds[:, 1].min()

    # bounds[:, 0] += x_shift
    # bounds[:, 1] += y_shift
    bounds = np.flip(bounds, axis=1)  # flip x, y to r, c

    return bounds, x_shift, y_shift


def inverse_warp(img, h_matrix) -> np.ndarray:
    assert h_matrix.shape == (3, 3)

    h, w, c = img.shape
    H, W = range(h), range(w)

    # initialize warped img matrix
    box, x_shift, y_shift = bounding_box(img, h_matrix)
    row_shift, col_shift = y_shift, x_shift

    bound_rows, bound_cols = box[:, 0], box[:, 1]
    warp_h, warp_w = bound_rows.max() + 1, bound_cols.max() + 1
    # + 1 since the bounding box needs to be a valid index

    hr, wr = 1, 1
    warp_h, warp_w = int(np.ceil(warp_h * hr)), int(np.ceil(warp_w * wr))
    warped = np.zeros((warp_h, warp_w, c))

    # compute target coordinates
    print("====target====")
    target_rr, target_cc = sk.draw.polygon(
        bound_rows * hr, bound_cols * wr, shape=(warp_h, warp_w)
    )

    # reverse shifting to get the original trasformed values
    target_rr -= row_shift
    target_cc -= col_shift

    print(warped.shape)
    print(target_rr.min(), target_cc.min())
    print(target_rr.max(), target_cc.max())

    # compute source coordinates
    print("=====src=====")

    num_pts = len(target_rr)
    target_pts = np.vstack((target_cc, target_rr, np.ones((1, num_pts))))

    src_pts = np.linalg.inv(h_matrix) @ target_pts
    src_cc, src_rr = src_pts.T[:, 0], src_pts.T[:, 1]

    # make sure indices are integers
    src_rr = np.int32(np.round(src_rr))
    src_cc = np.int32(np.round(src_cc))
    # shift indices to zero-indexed
    # src_rr += row_shift
    # src_cc += col_shift
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

    return warped, [col_shift, row_shift]
