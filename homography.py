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
            row1 = [x1, x2, 1, 0, 0, 0, -x1 * x2, -y1 * x2]
            row2 = [0, 0, 0, x1, x2, 1, -x1 * y2, -y1 * y2]
            A.append(row1)
            A.append(row2)

    A = np.array(A)
    B = [x for x, y in im2_pts] + [y for x, y in im2_pts]
    B = np.array(B)
    print(A.shape, num_pts)
    print(B.shape)
 
    params, *_ = np.linalg.lstsq(A, B)
    a, b, c, d, e, f, g, h = params
    H = np.array([[a, b, c], [d, e, f], [g, h, 1]])
    return H

def target_pts(pts, H):
    """ Returns target points in forward warping """
    pts_3D = [[x, y, 1] for x, y in pts]
    pts_3D = np.array(pts_3D).T
    target = (H @ pts_3D).T
    return target[:, :2] # remove 3rd dimension

def warp(img, h_matrix) -> np.ndarray:
#     assert_img_type(img)
#     assert_points(img_pts)
#     assert_points(target_pts)

    h, w, c = img.shape
    H,W=range(h),range(w)
    print(f'{H = }')
#     warped = np.zeros_like(img)
    warped = np.zeros((int(h*1.5),int(w*1.5), c))

    # Interpolation functions
    f_red, f_green, f_blue = [interpolate.RectBivariateSpline(
        H, W, img[:, :, c]) for c in range(3)]

#     coordinates = np.meshgrid(H,W) # for each pixel
    coordinates = np.array(list(itertools.product(H, W)))
    print(coordinates.shape)
    
    pts_3D = [[x, y, 1] for x, y in coordinates]
    pts_3D = np.array(pts_3D).T
    print(pts_3D.shape)
    
    target_pts = h_matrix @ pts_3D
    print(target_pts.shape)

    # fix w
#     target_pts[0] /= target_pts[2] # fix all x
#     target_pts[1] /= target_pts[2] # fix all y
    target_pts /= target_pts[2]

    target_cc, target_rr = target_pts[:2] # x, y = c, r
    src_cc, src_rr = coordinates[:, 0], coordinates[:, 1]
    
    target_cc = np.int32(np.round(target_cc))
    target_rr = np.int32(np.round(target_rr))

    # Interpolate
    warped[target_cc, target_rr, 0] = f_red.ev(src_cc, src_rr)
    warped[target_cc, target_rr, 1] = f_green.ev(src_cc, src_rr)
    warped[target_cc, target_rr, 2] = f_blue.ev(src_cc, src_rr)

    warped = np.clip(warped, 0.0, 1.0)
    assert_img_type(warped)
    return warpeds