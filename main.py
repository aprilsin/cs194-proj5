import argparse
import os
import sys
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
    dest="save_data",
    default=False,
    action="store_true",
    help="Whether intermediate data will be saved.",
)
parser.add_argument(
    "-l",
    "--load",
    dest="load_data",
    default=False,
    action="store_true",
    help="Whether intermediate data will be saved.",
)

args = parser.parse_args()
args.images = [Path(x) for x in args.images]

SAVE = args.save_data
LOAD = args.load_data


def manual_stitch_plane():
    i, j = [im_name.stem[-1] for im_name in args.images]
    imgs = [utils.read_img(im) for im in args.images]
    pts = [utils.pick_points(im, NUM_PTS) for im in imgs]

    h, w, c = imgs[0].shape
    num_pixels = 1600 * 1600
    RESIZE = h * w > num_pixels
    r = int(h * w / num_pixels)
    if RESIZE and r > 1:
        print("Resizing images")
        imgs = [
            sk.transform.resize(
                im, (im.shape[0] // r, im.shape[1] // r), anti_aliasing=True
            )
            for im in imgs
        ]
        pts = [p / r for p in pts]

    im1, im2 = imgs
    pts1, pts2 = pts
    plane_pts = (pts1 + pts2) / 2

    # warp image 1
    H1 = homography.homo_matrix(pts1, plane_pts)
    warp1, shift1 = homography.inverse_warp(im1, H1)
    warp_pts1 = homography.warp_pts(pts1, H1, shift1)
    # warp image 2
    H2 = homography.homo_matrix(pts2, plane_pts)
    warp2, shift2 = homography.inverse_warp(im2, H2)
    warp_pts2 = homography.warp_pts(pts2, H2, shift2)

    aligned1, aligned2 = rectification.align(warp1, warp2, warp_pts1, warp_pts2)
    blended = rectification.blend(aligned1, aligned2, method=BLEND_METHOD)

    mosaic_name = OUTDIR_1 / (name + "_mosaic.jpg")
    plt.imsave(mosaic_name, blended)
    print(f"Mosaic saved as {mosaic_name}")
    return


def manual_stitch_direct():
    imgs = [utils.read_img(im) for im in args.images]

    if LOAD:
        try:
            pts1 = utils.load_points(args.images[0])
            name_a = args.images[1].stem + "a" + args.images[1].suffix
            name_b = args.images[1].stem + "b" + args.images[1].suffix
            pts2a = utils.load_points(args.images[1].parent / name_a)
            pts2b = utils.load_points(args.images[1].parent / name_b)
            pts3 = utils.load_points(args.images[2])
            pts = [pts1, pts2a, pts2b, pts3]
            print("Loaded points for 3 images")
        except:
            raise FileExistsError()
    else:
        pts1 = utils.pick_points(args.images[0], NUM_PTS)
        pts2a = utils.pick_points(args.images[1], NUM_PTS)
        pts2b = utils.pick_points(args.images[1], NUM_PTS)
        pts3 = utils.pick_points(args.images[2], NUM_PTS)
        pts = [pts1, pts2a, pts2b, pts3]
        if SAVE:
            print("Saving points")
            pts1 = utils.save_points(pts1, args.images[0])
            name_a = args.images[1].stem + "a" + args.images[1].suffix
            name_b = args.images[1].stem + "b" + args.images[1].suffix
            pts2a = utils.save_points(pts2a, args.images[1].parent / name_a)
            pts2b = utils.save_points(pts2b, args.images[1].parent / name_b)
            pts3 = utils.save_points(pts3, args.images[2])

    h, w, c = imgs[0].shape
    num_pixels = 1600 * 1600
    RESIZE = h * w > num_pixels
    r = int(h * w / num_pixels)
    if RESIZE and r > 1:
        print(f"Resizing images")
        for i in range(3):
            imgs[i] = sk.transform.resize(
                imgs[i],
                (imgs[i].shape[0] // r, imgs[i].shape[1] // r),
                anti_aliasing=True,
            )
        for i in range(4):
            pts[i] /= r

    im1, im2, im3 = imgs
    pts1, pts2a, pts2b, pts3 = pts

    # warp image 1
    print("Warp image 1 to image 2")
    H1 = homography.homo_matrix(pts1, pts2)
    warp1, shift1 = homography.inverse_warp(im1, H1)
    warp1_pts = homography.warp_pts(pts1, H1, shift1)

    # no need to warp image 2
    warp2, warp2_pts = im2, pts2

    print("Align and blend image 1 and 2")
    aligned1, aligned2, _, blend_pts = rectification.align(
        warp1, warp2, warp1_pts, warp2_pts
    )
    blend_12 = rectification.blend(aligned1, aligned2, method=BLEND_METHOD)

    ### merge im2 and im3 ###
    print("Warp image 3 to image 2")
    H3 = homography.homo_matrix(pts3, blend_pts)
    warp3, shift3 = homography.inverse_warp(im3, H3)
    warp3_pts = homography.warp_pts(pts3, H3, shift3)

    print("Align and blend with image 3")
    aligned12, aligned3, _, _ = rectification.align(
        blend_12, warp3, blend_pts, warp3_pts
    )
    blend_123 = rectification.blend(aligned12, aligned3, method=BLEND_METHOD)

    if SAVE:
        print("Saving images")
        plt.imsave(OUTDIR_1 / (name + "_warp1.jpg"), warp1)
        plt.imsave(OUTDIR_1 / (name + "_blend12.jpg"), blend_12)
        plt.imsave(OUTDIR_1 / (name + "_warp3.jpg"), warp3)

    mosaic_name = OUTDIR_1 / (name + "_mosaic.jpg")
    plt.imsave(mosaic_name, blend_123)
    print(f"Mosaic saved as {mosaic_name}")
    return


def auto_stitching():
    pass


if __name__ == "__main__":

    num_imgs = len(args.images)
    name = (Path(args.images[0]).stem)[:-1]

    print(
        f"""
stitching {num_imgs} images for {name}...
detection: {args.detection}"""
    )

    # set up file names
    # (DATA / name).mkdir(parents=False, exist_ok=True)

    args.images = [Path(x) for x in args.images]
    if args.detection == "manual":
        if num_imgs == 2:  # warp images to a frontal plane
            print("rectification: frontal plane\n")
            manual_stitch_plane()
            sys.exit()
        elif num_imgs == 3:  # warp image 1 and image 3 to middle image 2
            print("rectification: warp to middle image\n")
            manual_stitch_direct()
            sys.exit()
        else:
            raise ValueError(f"{num_imgs = }")
    if args.detection == "auto":
        print("not implemented yet")
        sys.exit()
