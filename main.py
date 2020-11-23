import argparse
import sys
import os
import homography, rectification
from pathlib import Path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as io

import filters
import homography
import rectification
import utils
from constants import *

# class ToPath(argparse.Action):
#     def __call__(self, parser, namespace, values, option_string=None)

parser = argparse.ArgumentParser()

parser.add_argument(
    "detection",
    type=str,
    choices=["manual", "auto"],
    default="manual",
    help="Choose manual or auto stitching",
)
# parser.add_argument("stitch", type=str, choices=["plane", "merge"])
parser.add_argument("images", nargs="+")  # TODO: action=ToPath
parser.add_argument(
    "-s",
    "--save",
    type=bool,
    default=False,
    help="Whether intermediate data will be saved.",
)

args = parser.parse_args()
args.images = [Path(x) for x in args.images]


def manual_stitch_plane():
    i, j = [im_name.stem[-1] for im_name in args.images]
    imgs = [utils.read_img(im) for im in args.images]
    pts = [utils.pick_points(im, 4) for im in imgs]

    h, w, c = imgs[0].shape
    num_pixels = 1600 * 1600
    RESIZE = h * w > num_pixels
    r = int(h * w / num_pixels)
    if RESIZE and r > 1:
        print("resizing to target shape ({h / r}, {w / r}")
        imgs = [
            sk.transform.resize(
                im, (im.shape[0] // r, im.shape[1] // r), anti_aliasing=True
            )
            for im in imgs
        ]
        pts = [p / r for p in pts]

    im1, im2 = imgs
    im1_pts, im2_pts = pts
    plane_pts = (im1_pts + im2_pts) / 2

    # warp image 1
    h_matrix1 = homography.homo_matrix(im1_pts, plane_pts)
    warp1, shift_pts1 = homography.inverse_warp(im1, h_matrix1)
    warp_pts1 = homography.warp_pts(im1_pts, h_matrix1, shift_pts1)
    # warp image 2
    h_matrix2 = homography.homo_matrix(im2_pts, plane_pts)
    warp2, shift_pts2 = homography.inverse_warp(im2, h_matrix2)
    warp_pts2 = homography.warp_pts(im2_pts, h_matrix2, shift_pts2)

    aligned1, aligned2 = rectification.align(warp1, warp2, warp_pts1, warp_pts2)
    blended = rectification.blend(aligned1, aligned2, BLEND_METHOD)

    mosaic_name = OUTDIR / (name + "_mosaic.jpg")
    plt.imsave(mosaic_name, blended)
    return


def manual_stitch_direct():
    imgs = [utils.read_img(im) for im in args.images]

    for i in range(len(imgs)):
        h, w, c = imgs[0].shape
        num_pixels = 1600 * 1600
        RESIZE = h * w > num_pixels
        r = int(h * w / num_pixels)
        if RESIZE and r > 1:
            imgs[i] = sk.transform.resize(
                imgs[i],
                (imgs[i].shape[0] // r, imgs[i].shape[1] // r),
                anti_aliasing=True,
            )
    im1, im2, im3 = imgs

    # merge im1 and im2
    im1_pts = utils.pick_points(im1, 4)
    im2_pts = utils.pick_points(im2, 4)

    # warp image 1
    h_matrix1 = homography.homo_matrix(im1_pts, im2_pts)
    warp1, shift_pts1 = homography.inverse_warp(im1, h_matrix1)
    warp_pts1 = homography.warp_pts(im1_pts, h_matrix1, shift_pts1)

    # warp image 2
    im2_pts = utils.pick_points(im2, 4)
    im3_pts = utils.pick_points(im3, 4)

    h_matrix3 = homography.homo_matrix(im3_pts, im2_pts)
    warp3, shift_pts3 = homography.inverse_warp(im3, h_matrix3)
    warp_pts3 = homography.warp_pts(im2_pts, h_matrix3, shift_pts3)

    aligned1, aligned2 = rectification.align(warp1, im2, warp_pts1, warp_pts2)
    blended = rectification.two_band_blend(aligned1, aligned2)

    mosaic_name = OUTDIR / (name + "_mosaic.jpg")
    plt.imsave(mosaic_name, blended)
    pass


def auto_stitching():
    pass


if __name__ == "__main__":

    num_imgs = len(args.images)
    name = (Path(args.images[0]).stem)[:-1]

    print(
        f"stitching {num_imgs} images for {name}...\ndetection-method = {args.detection}"
    )

    # set up file names
    DATA = Path("data")
    OUTDIR = Path("output")
    assert DATA.exists()
    assert OUTDIR.exists()
    (DATA / name).mkdir(parents=False, exist_ok=True)

    args.images = [Path(x) for x in args.images]
    if args.detection == "manual":
        if num_imgs == 2:  # warp images to a frontal plane
            print("rectification method = frontal plane")
            manual_stitch_plane()
            sys.exit()
        elif num_imgs == 3:  # warp image 1 and image 3 to middle image 2
            print("rectification method = warp to middle image")
            manual_stitch_direct()
            sys.exit()
        else:
            raise ValueError(f"{num_imgs = }")