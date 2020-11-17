import numpy as np
from my_types import assert_points
import itertools
from scipy import interpolate

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

def target_pts(pts, H):
    """ Returns target points in forward warping """
    pts_3D = [[x, y, 1] for x, y in pts]
    pts_3D = np.array(pts_3D).T
    target = (H @ pts_3D).T
    return target[:, :2] # remove 3rd dimension

def warp(img, h_matrix) -> np.ndarray:
#     assert_img_type(img)
    assert h_matrix.shape == (3, 3)

    h, w, c = img.shape
    H,W=range(h),range(w)
    
    # initialize warped img matrix
    r = 1.8
    warped = np.zeros((int(h*r),int(w*r), c))
    print(img.shape)
    print(warped.shape)
    
    # interpolation functions
    f_red, f_green, f_blue = [interpolate.RectBivariateSpline(
        H, W, img[:, :, c]) for c in range(3)]

    # source coordinates
#     coordinates = np.meshgrid(H,W) # for each pixel
    coordinates = np.array(list(itertools.product(H, W)))
    src_rr, src_cc = coordinates[:, 0], coordinates[:, 1]
    
    pts_3D = [[x, y, 1] for x, y in coordinates]
    pts_3D = np.array(pts_3D).T # transpose so that each column is [x, y, 1]

    # compute target coordinates
    target_pts = h_matrix @ pts_3D
    # fix w
    target_pts /= target_pts[2]
    
    print("=====src=====")
    print(img.shape)
    print(src_rr.min(), src_cc.min())
    print(src_rr.max(), src_cc.max())
    
    target_cc, target_rr = target_pts[:2] # x, y = c, r
    
    target_rr = np.int32(np.round(target_rr))
    target_cc = np.int32(np.round(target_cc))
    if target_rr.min() < 0:
        target_rr += abs(target_rr.min())
    if target_cc.min() < 0:
        target_cc += abs(target_cc.min())
    
    print("====target====")
    print(warped.shape)
    print(target_rr.min(), target_cc.min())
    print(target_rr.max(), target_cc.max())

    # Interpolate
    print(target_rr)
    print(target_cc)
    for i, f in enumerate([f_red, f_green, f_blue]):
        warped[target_rr, target_cc, i] = f.ev(xi=src_cc, yi=src_rr) # x = cols, y = rows

    warped = np.clip(warped, 0.0, 1.0)
#     assert_img_type(warped)
    return warped
