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
    warped = np.zeros((int(h*1.2),int(w*1.2), c))

    # Interpolation functions
    f_red, f_green, f_blue = [interpolate.RectBivariateSpline(
        H, W, img[:, :, c]) for c in range(3)]

#     coordinates = np.meshgrid(H,W) # for each pixel
    coordinates = np.array(list(itertools.product(H, W)))
    
    pts_3D = [[x, y, 1] for x, y in coordinates]
    pts_3D = np.array(pts_3D).T
    target_pts = h_matrix @ pts_3D
    
    rows = pts_3D.T[:, 0]
    cols = pts_3D.T[:, 1]
    print(warped.shape)
    print(rows.min(), cols.min())
    print(rows.max(), cols.max())
    
    # fix w
    target_pts /= target_pts[2]
    
    rows = target_pts[:, 0]
    cols = target_pts[:, 1]
    print(warped.shape)
    print(rows.min(), cols.min())
    print(rows.max(), cols.max())
    
    target_cc, target_rr = target_pts[:2] # x, y = c, r
    # normalize
#     target_rr += abs(rows.min())
#     target_cc += abs(cols.min())
    src_rr, src_cc = coordinates[:, 0], coordinates[:, 1]
    
    print(img.shape)
    print(src_rr.max(), src_cc.max())
    
    target_cc = np.int32(np.round(target_cc))
    target_rr = np.int32(np.round(target_rr))

    # Interpolate
    for i, f in enumerate([f_red, f_green, f_blue]):
        warped[target_cc, target_rr, i] = f.ev(xi=src_cc, yi=src_rr) # x = cols, y = rows

#     warped = np.clip(warped, 0.0, 1.0)
#     assert_img_type(warped)
    return warped