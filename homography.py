import numpy as np
from my_types import assert_points


def homo_matrix(im1_pts: np.ndarray, im2_pts: np.ndarray):
    """Returns a homographic transformation matrix from ptsA to ptsB"""
    # assert_points(im1_pts, ratio=False)
    # assert_points(im2_pts, ratio=False)
    assert len(im1_pts) == len(im2_pts)

    num_pts = len(im1_pts)

    A = []
    B = []
    for x1, y1 in im1_pts:
        for x2, y2 in im2_pts:
            row1 = [x1, x2, 1, 0, 0, 0, -x1 * x2, -y1 * x2]
            row2 = [0, 0, 0, x1, x2, 1, -x1 * y2, -y1 * y2]
            A.append(row1)
            A.append(row2)
    A = np.array(A)
    B = [x for x, y in im2_pts] + [y for x, y in im2_pts]
    B = np.array(B).T
    print(A.shape, num_pts)
    print(B.shape)

    params = np.linalg.solve(A, B)
    print(params)
    a, b, c, d, e, f, g, h = params
    H = np.array([a, b, c], [d, e, f], [g, h, 1])
    return H
