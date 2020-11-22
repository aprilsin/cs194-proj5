import numpy as np
from scipy import signal

import filters


def match_shift(im1, im2, pts1, pts2):
    pts1 = np.int32(np.round(pts1))
    pts2 = np.int32(np.round(pts2))
    aligned_pts1, aligned_pts2 = pts1.copy(), pts2.copy()

    # w, h = x, y
    p1_w, p1_h = pts1[:, 0].max(), pts1[:, 1].max()
    p2_w, p2_h = pts2[:, 0].max(), pts2[:, 1].max()

    im1_pad, im2_pad = [[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]

    h_diff, w_diff = abs(p1_h - p2_h), abs(p1_w - p2_w)
    if p1_h < p2_h:
        im1_pad[0][0] = h_diff  # pad before
        #         im2_pad[0][1] = h_diff  # pad after
    else:
        im2_pad[0][0] = h_diff  # pad before
        #         im1_pad[0][1] = h_diff  # pad after

    if p1_w < p2_w:
        im1_pad[1][0] = w_diff  # pad before
        #         im2_pad[1][1] = w_diff  # pad after
    else:
        im2_pad[1][0] = w_diff  # pad before
        #         im1_pad[1][1] = w_diff  # pad after

    im1_pad = tuple((before, after) for before, after in im1_pad)
    im2_pad = tuple((before, after) for before, after in im2_pad)
    im1_shifted = np.pad(im1, im1_pad)
    im2_shifted = np.pad(im2, im2_pad)

    return im1_shifted, im2_shifted


def match_shape(im1, im2):
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    assert c1 == c2, (im1.shape, im2.shape)

    im1_pad, im2_pad = [[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]

    h_diff, w_diff = abs(h1 - h2), abs(w1 - w2)

    if h1 < h2:
        im1_pad[0][1] = h_diff  # pad after
    else:
        im2_pad[0][1] = h_diff  # pad after

    if w1 < w2:
        im1_pad[1][1] = w_diff  # pad after
    else:
        im2_pad[1][1] = w_diff  # pad after

    im1_pad = tuple((before, after) for before, after in im1_pad)
    im2_pad = tuple((before, after) for before, after in im2_pad)
    im1_matched = np.pad(im1, im1_pad)
    im2_matched = np.pad(im2, im2_pad)

    assert im1_matched.shape == im2_matched.shape, (
        im1_matched.shape,
        im2_matched.shape,
    )
    return im1_matched, im2_matched


def align(warped_im1, warped_im2, warped_im1_pts, warped_im2_pts):
    """
    Returns two separate images with padding added for alignment.
    Input points should be standardized. (using the same coordinate system / warped)
    """
    im1, im2, pts1, pts2 = warped_im1, warped_im2, warped_im1_pts, warped_im2_pts

    im1, im2 = match_shift(im1, im2, pts1, pts2)
    im1, im2 = match_shape(im1, im2)

    return im1, im2


def overlap_mask(im1, im2):
    assert im1.shape == im2.shape, (im1.shape, im2.shape)
    tmp1 = np.where(im1 != 0, True, False)
    tmp2 = np.where(im2 != 0, True, False)
    return (tmp1 & tmp2).astype(int).astype(np.float64)


def alpha_blend(im1, im2):
    mask_hard = overlap_mask(im1, im2)
    mask_soft = filters.gauss_blur(mask_hard)
    base = (1 - mask_hard) * (im1 + im2)
    blend = mask_soft * (im1 * 0.5 + im2 * 0.5)
    return np.clip(np.add(base, blend), 0.0, 1.0)


def average_blend(im1, im2):
    overlap = overlap_mask(im1, im2)
    base = (1 - overlap) * (im1 + im2)
    blend = overlap * (im1 * 0.5 + im2 * 0.5)
    return np.clip(np.add(base, blend), 0.0, 1.0)


def two_band_blend(im1, im2):
    assert im1.shape == im2.shape

    overlap = overlap_mask(im1, im2)
    base = (1 - overlap) * (im1 + im2)

    mask_soft = filters.gauss_blur(overlap)

    low1 = filters.gauss_blur(im1, display=True)
    low2 = filters.gauss_blur(im2, display=True)

    low = mask_soft * (low1 * 0.5 + low2 * 0.5)

    high1 = im1 - low1
    high2 = im2 - low2
    high = overlap * (high1 + high2)

    result = low + high + base
    return np.clip(result, 0.0, 1.0)


def blend(im1, im2, method="two-band"):
    """ Blend two images together """
    if method == "two-band":
        return two_band_blend(im1, im2)
    else:
        return None


# blends stacks for one channel
def blend_channel(im1_Lstack, im2_Lstack, region_stack, mask):
    assert len(im1_Lstack) == len(im2_Lstack) == len(region_stack)

    blended = []
    for L1, L2, R in zip(im1_Lstack, im2_Lstack, region_stack):
        R = signal.convolve2d(R, mask, mode="same")
        curr_level = R * L1 + (1 - R) * L2
        blended.append(curr_level)
    blended = np.array(blended)
    result = np.sum(blended, axis=0)
    print(result.shape)
    return result


def stitch(im1, im2, im1_pts, im2_pts):
    """ Stictch two warped images. All inputs should be warped. """

    align1, align2 = align(im1, im2, im1_pts, im2_pts)
    mosaic = alpha_blend(im1, im2)
    return mosaic
