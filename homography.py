import numpy as np
from my_types import assert_points
import itertools
from scipy import interpolate
import skimage as sk

def homo_matrix(im1_pts: np.ndarray, im2_pts: np.ndarray):
    """Returns a homographic transformation matrix from ptsA to ptsB"""
    # assert_points(im1_pts, ratio=False)
    # assert_points(im2_pts, ratio=False)
    assert len(im1_pts) == len(im2_pts)

    num_pts = len(im1_pts)

    A = []
    B = []
    for (x1, y1), (x2, y2) in zip(im1_pts, im2_pts):
            row1 = [x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2]
            row2 = [0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2]
            A.append(row1)
            A.append(row2)
            B.extend((x2, y2))

    A = np.array(A)
    B = np.array(B)
    print(f'{num_pts = }')
    print(A.shape)
    print(B.shape)
 
    params, *_ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c, d, e, f, g, h = params
    H = np.array([[a, b, c], [d, e, f], [g, h, 1]])
#     H = H/i
    return H

def bounding_box(img, H):
    """ Return the corners of the projected image in H, W domain """
    h, w, c = img.shape
    corners_hw = [[0, 0], [0, w], [h, 0], [h, w]] # TODO should there be a minus 1?
    corners_xy = [[c, r] for r, c in corners_hw] # x, y = c, r
    print(corners_xy)
    
    pts = [[x, y, 1] for x, y in corners_xy]
    pts = np.array(pts).T

    bounds = H @ pts
    
    bounds /= bounds[2] # fix w, scaling due to transformation
    bounds = bounds.T[:, :2] # reshape into the form [[x1, y1], [x2, y2]]
    
    # make sure indicies are integers
    bounds = np.ceil(bounds).astype(np.int64)
    
    # shift indices to zero-indexed    
    min_x = bounds[:, 0].min()
    min_y = bounds[:, 1].min()
    bounds[:, 0] += -min_x
    bounds[:, 1] += -min_y
    bounds = np.flip(bounds, axis=1) # flip x, y to r, c
    row_shift = -min_y
    col_shift = -min_x
    return bounds, row_shift, col_shift
#     return bounds


def empty_warp(img, h_matrix, r):
    h, w, c = img.shape
    box, _, _ = bounding_box(img, h_matrix)
    
    warp_h, warp_w = box[:, 0].max() + 1, box[:, 1].max() + 1
    # + 1 since the bounding box needs to be a valid index
    
    warp_h, warp_w = int(warp_h * r), int(warp_w * r)
    return np.zeros((warp_h,warp_w, c))
    
    
def forward_warp(img, h_matrix, r) -> np.ndarray:
#     assert_img_type(img)
    assert h_matrix.shape == (3, 3)

    h, w, c = img.shape
    H,W=range(h),range(w)
    
    # initialize warped img matrix
    warped = empty_warp(img, h_matrix, r)
    
    # interpolation functions
    f_red, f_green, f_blue = [interpolate.RectBivariateSpline(
        H, W, img[:, :, c]) for c in range(3)]

    # compute source coordinates
#     coordinates = np.meshgrid(H,W) # for each pixel
    coordinates = np.array(list(itertools.product(H, W)))
    src_rr, src_cc = coordinates[:, 0], coordinates[:, 1]
    
    print("=====src=====")
    print(img.shape)
    print(src_rr.min(), src_cc.min())
    print(src_rr.max(), src_cc.max())

    # compute target coordinates
    pts_3D = [[c, r, 1] for r, c in coordinates] # x, y = c, r
    pts_3D = np.array(pts_3D).T # transpose so that each column is [x, y, 1]

    target_pts = h_matrix @ pts_3D
    target_pts /= target_pts[2] # fix w
    print(f'{target_pts.shape = }')
    
    target_cc, target_rr = target_pts.T[:, 0], target_pts.T[:, 1] # x, y = c, r
#     target_cc, target_rr = target_pts[0, :], target_pts[1, :] # x, y = c, r
#     print(target_rr.shape, target_cc.shape)
    
    # make sure indicies are integers
    target_rr = np.int32(np.round(target_rr))
    target_cc = np.int32(np.round(target_cc))
    # shift indices to zero-indexed
    target_rr += -target_rr.min()
    target_cc += -target_cc.min()
    
    print("====target====")
    print(warped.shape)
    print(target_rr.min(), target_cc.min())
    print(target_rr.max(), target_cc.max())
#     print(target_rr.argmin(), target_cc.argmin())
#     print(target_rr.argmax(), target_cc.argmax())
#     print(coordinates[target_rr.argmax()])
#     print(coordinates[target_cc.argmax()])

    # Interpolate
    print("=====interpolate=====")
    print(target_rr)
    print(target_cc)
    for i, f in enumerate([f_red, f_green, f_blue]):
        warped[target_rr, target_cc, i] = f.ev(xi=src_cc, yi=src_rr) # x = cols, y = rows

    warped = np.clip(warped, 0.0, 1.0)
#     assert_img_type(warped)
    return warped


def inverse_warp(img, h_matrix) -> np.ndarray:
#     assert_img_type(img)
    assert h_matrix.shape == (3, 3)

    h, w, c = img.shape
    H,W=range(h),range(w)
    
    # initialize warped img matrix
    box, row_shift, col_shift = bounding_box(img, h_matrix)
    bound_rows = box[:, 0]
    bound_cols = box[:, 1]
    warp_h, warp_w = bound_rows.max() + 1, bound_cols.max() + 1
    # + 1 since the bounding box needs to be a valid index
    warped = np.zeros((warp_h, warp_w, c))
    
    # compute target coordinates
    print("====target====")
    target_rr, target_cc = sk.draw.polygon(bound_rows, bound_cols, shape=(warp_h, warp_w))
    
    print(warped.shape)
    print(target_rr.min(), target_cc.min())
    print(target_rr.max(), target_cc.max())
    
    # compute source coordinates
    print("=====src=====")
    num_pts = len(target_rr)
    target_pts = np.vstack((target_cc, target_rr, np.ones((1, num_pts))))

    src_pts = np.linalg.inv(h_matrix) @ target_pts
    src_cc, src_rr = src_pts.T[:, 0], src_pts.T[:, 1] # x, y = c, r
    print(src_rr.shape, src_cc.shape)
    
    # make sure indicies are integers
    src_rr = np.int32(np.round(src_rr))
    src_cc = np.int32(np.round(src_cc))
    # shift indices to zero-indexed
#     src_rr += -src_rr.min()
#     src_cc += -src_cc.min()
    
    print(img.shape)
    print(src_rr.min(), src_cc.min())
    print(src_rr.max(), src_cc.max())
#     print(src_rr.argmin(), src_cc.argmin())
#     print(src_rr.argmax(), src_cc.argmax())
#     print(coordinates[src_rr.argmax()])
#     print(coordinates[src_cc.argmax()])

    # interpolate
    print("=====interpolate=====")
    f_red, f_green, f_blue = [interpolate.RectBivariateSpline(
        H, W, img[:, :, c]) for c in range(3)]
    
    for i, f in enumerate([f_red, f_green, f_blue]):
        warped[target_rr, target_cc, i] = f.ev(xi=src_cc, yi=src_rr)
        
    return warped
    