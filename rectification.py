import numpy as np

import filters


def blend(im1, im2, method):
    """ Blend two images together """
    if method == "two-band":
        return two_band_blend(im1, im2)
    else:
        return None


def two_band_blend(im1, im2):
    assert im1.shape == im2.shape
    low1 = filters.gaussian_filter(im1)
    low2 = filters.gaussian_filter(im2)
    high1 = filters.unsharp_mask_filter(im1)
    high2 = filters.unsharp_mask_filter(im2)
    return np.sum([low1, low2, high1, high2])

def align(warped_im1, warped_im2, warped_im1_pts, warped_im2_pts):
    """ 
    Returns two separate images with padding added for alignment.
    Input points should be standardized. (using the same coordinate system / warped)
    """
    im1, im2 = warped_im1, warped_im2
    im1_pts = np.int32(np.round(warped_im1_pts))
    im2_pts = np.int32(np.round(warped_im2_pts))
    aligned_pts1, aligned_pts2 = im1_pts.copy(), im2_pts.copy()

    # w, h = x, y
    p1_w, p1_h = im1_pts[:, 0].max(), im1_pts[:, 1].max()
    p2_w, p2_h = im2_pts[:, 0].max(), im2_pts[:, 1].max()

    im1_pad, im2_pad = [[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]

    h_diff, w_diff = abs(p1_h - p2_h), abs(p1_w - p2_w)
    if p1_h < p2_h:
        im1_pad[0][0] = h_diff  # pad before
#         im2_pad[0][1] = h_diff  # pad after
        algined_pts1[:, 1] += h_diff
    else:
        im2_pad[0][0] = h_diff  # pad before
#         im1_pad[0][1] = h_diff  # pad after
        aligned_pts2[:, 1] += h_diff

    if p1_w < p2_w:
        im1_pad[1][0] = w_diff  # pad before
#         im2_pad[1][1] = w_diff  # pad after
        algined_pts1[:, 0] += w_diff
    else:
        im2_pad[1][0] = w_diff  # pad before
#         im1_pad[1][1] = w_diff  # pad after
        aligned_pts2[:, 0] += w_diff
    
    im1_pad = tuple((before, after) for before, after in im1_pad)
    im2_pad = tuple((before, after) for before, after in im2_pad)
    im1_aligned = np.pad(im1, im1_pad)
    im2_aligned = np.pad(im2, im2_pad)

    assert im1_aligned.shape == im2_aligned.shape, f"{h_diff = }, {w_diff = }\n{im1.shape} -> {im1_aligned.shape} \n {im2.shape} -> {im2_aligned.shape}"
    return im1_aligned, im2_aligned


def blend_windows(aligned1_pts, aligned2_pts):
    x, y = aligned1_pts[:, 0], aligned2_pts[:, 1]


def stitch(im1, im2, im1_pts, im2_pts, blend_method="two-band"):
    """ Stictch two warped images. All inputs should be warped. """
    im1, im2 = align(im1, im2, im1_pts, im2_pts)
    mosaic = blend(im1, im2, blend_method)
    return mosaic
