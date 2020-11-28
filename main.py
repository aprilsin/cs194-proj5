import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as io

# import constants
import constants
import descriptor
import detector
import filters
import homography
import matching
import rectification
import utils
from constants import DATA, OUTDIR_1, OUTDIR_2
import math

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
parser.add_argument(
    "images",
    nargs="+",
    help="2 or 3 images to be stitched in entered in stitching order",
)  # TODO: action=ToPath

parser.add_argument(
    "-s",
    "--save",
    dest="save_data",
    action="store_false",
    help="Save intermediate data.",
)

parser.add_argument(
    "-S",
    "--show",
    dest="show_plots",
    action="store_true",
    help="Show intermediate images.",
)

parser.add_argument(
    "-l",
    "--load",
    dest="load_data",
    action="store_true",
    help="Load points from pickle files.",
)

parser.add_argument(
    "-d",
    "--debug",
    dest="debug",
    action="store_true",
    help="Show debugging print statements.",
)

args = parser.parse_args()
args.images = [Path(x) for x in args.images]

constants.SAVE = args.save_data
constants.LOAD = args.load_data
constants.DEBUG = args.debug
constants.SHOW = args.show_plots


def manual_stitch_plane():
    i, j = [im_name.stem[-1] for im_name in args.images]
    imgs = [utils.read_img(im, resize=False) for im in args.images]

    if constants.LOAD:
        try:
            pts1 = utils.load_points(args.images[0])
            pts2 = utils.load_points(args.images[1])
            pts = [pts1, pts2]
            print("Loaded points for 2 images")
        except:
            raise FileExistsError()
    else:
        pts = [utils.pick_points(im, constants.NUM_PTS) for im in args.images]
        if constants.SAVE:
            print("Saving points")
            utils.save_points(pts1, args.images[0])
            utils.save_points(pts2, args.images[1])

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

    aligned1, aligned2, *_ = rectification.align(warp1, warp2, warp_pts1, warp_pts2)
    blended = rectification.blend(aligned1, aligned2, method=constants.BLEND_METHOD)

    mosaic_name = OUTDIR_1 / (name + "_mosaic.jpg")
    plt.imsave(mosaic_name, blended)
    print(f"Mosaic saved as {mosaic_name}")
    return


def manual_stitch_direct():
    imgs = [utils.read_img(im, resize=False) for im in args.images]

    if constants.LOAD:
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
        pts1 = utils.pick_points(args.images[0], constants.NUM_PTS)
        pts2a = utils.pick_points(args.images[1], constants.NUM_PTS)
        pts2b = utils.pick_points(args.images[1], constants.NUM_PTS)
        pts3 = utils.pick_points(args.images[2], constants.NUM_PTS)
        pts = [pts1, pts2a, pts2b, pts3]
        if constants.SAVE:
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
    H1 = homography.homo_matrix(pts1, pts2a)
    warp1, shift1 = homography.inverse_warp(im1, H1)
    warp1_pts = homography.warp_pts(pts1, H1, shift1)

    # no need to warp image 2
    warp2, warp2_pts = im2, pts2a

    print("Align and blend image 1 and 2")
    aligned1, aligned2, _, _, _, shift2 = rectification.align(
        warp1, warp2, warp1_pts, warp2_pts
    )
    blend_12 = rectification.blend(aligned1, aligned2, method=constants.BLEND_METHOD)

    ### merge im2 and im3 ###
    print("Warp image 3 to image 2")
    pts2b += shift2
    H3 = homography.homo_matrix(pts3, pts2b)
    warp3, shift3 = homography.inverse_warp(im3, H3)
    warp3_pts = homography.warp_pts(pts3, H3, shift3)

    print("Align and blend with image 3")
    aligned12, aligned3, *_ = rectification.align(blend_12, warp3, pts2b, warp3_pts)
    blend_123 = rectification.blend(aligned12, aligned3, method=constants.BLEND_METHOD)

    if constants.SAVE:
        print("Saving images")
        plt.imsave(OUTDIR_1 / (name + "_warp1.jpg"), warp1)
        plt.imsave(OUTDIR_1 / (name + "_blend12.jpg"), blend_12)
        plt.imsave(OUTDIR_1 / (name + "_warp3.jpg"), warp3)

    mosaic_name = OUTDIR_1 / (name + "_mosaic.jpg")
    plt.imsave(mosaic_name, blend_123)
    print(f"Mosaic saved as {mosaic_name}")
    return


def define_corners(im1, im2):

    print("====== CORNER DETECTION ======")

    strength1, coords1 = detector.get_harris(im1)
    strength2, coords2 = detector.get_harris(im2)
    print(f"Detected {len(coords1)} points from image 1.")
    print(f"Detected {len(coords2)} points from image 2.")

    constants.NUM_KEEP = min([constants.NUM_KEEP, len(coords1), len(coords2)])

    if constants.SAVE:
        fig = utils.plot_corners(im1, coords1)
        plt.savefig(OUTDIR_2 / (args.images[0].stem + "_detected.jpg"))
        fig = utils.plot_corners(im2, coords2)
        plt.savefig(OUTDIR_2 / (args.images[1].stem + "_detected.jpg"))

    print("====== ANMS ======")

    h, w = im1.shape
    constants.MAX_RADIUS = round(math.sqrt(h * w))
    corners1 = detector.anms(strength1, coords1)
    print(f"Selected top {len(corners1)} points from image 1.")

    h, w = im2.shape
    constants.MAX_RADIUS = round(math.sqrt(h * w))
    corners2 = detector.anms(strength2, coords2)
    print(f"Selected top {len(corners2)} points from image 2.")

    if constants.SAVE:
        fig = utils.plot_corners(im1, corners1)
        plt.savefig(OUTDIR_2 / (args.images[0].stem + "_anms.jpg"))
        fig = utils.plot_corners(im2, corners2)
        plt.savefig(OUTDIR_2 / (args.images[1].stem + "_anms.jpg"))

    print("====== CORNER DESCRIPTION ======")
    patches1 = descriptor.get_patches(im1, corners1)
    vectors1 = np.stack([p.flatten() for p in patches1])
    print(f"Computed descriptors of image 1.")
    patches2 = descriptor.get_patches(im2, corners2)
    vectors2 = np.stack([p.flatten() for p in patches2])
    print(f"Computed descriptors of image 2.")

    indices = np.random.randint(len(patches1), size=3)
    if constants.SAVE:
        for i in indices:
            fig = utils.plot_corners(im1, corners1)
            plt.savefig(OUTDIR_2 / (args.images[0].stem + f"_patch{i}.jpg"))
            fig = utils.plot_corners(im2, corners2)
            plt.savefig(OUTDIR_2 / (args.images[1].stem + f"_patch{i}.jpg"))

    print("====== CORNER MATCHING ======")
    matched1, matched2 = matching.match_features(corners1, vectors1, corners2, vectors2)

    print(f"Found {len(matched1)} candidate coorespondences.")

    if len(matched1) < 4:
        print(f"Cannot stitch images.")
        sys.exit()

    if constants.SAVE:
        fig = utils.plot_corners(im1, matched1, colors=constants.colors)
        plt.savefig(OUTDIR_2 / (args.images[0].stem + f"_match_{time.time():.0f}.jpg"))
        fig = utils.plot_corners(im2, matched2, colors=constants.colors)
        plt.savefig(OUTDIR_2 / (args.images[1].stem + f"_match_{time.time():.0f}.jpg"))

    # find best matches / inliers
    print("Do RANSAC...")
    result1, result2 = matching.ransac(matched1, matched2)

    print(f"Total features matched = {len(result1)}, {len(result2)}.")

    if constants.SAVE:
        fig = utils.plot_corners(im1, result1, colors=constants.colors)
        plt.savefig(OUTDIR_2 / (args.images[0].stem + f"_ransac_{time.time():.0f}.jpg"))
        fig = utils.plot_corners(im2, result2, colors=constants.colors)
        plt.savefig(OUTDIR_2 / (args.images[1].stem + f"_ransac_{time.time():.0f}.jpg"))

    return result1, result2


def stitch(im1, im2, pts1, pts2):
    print("===== STITCHING =====")

    # warp image 1
    print("Warp image 1 to image 2")
    H1 = homography.homo_matrix(pts1, pts2)
    warp1, shift1 = homography.inverse_warp(im1, H1)
    warp1_pts = homography.warp_pts(pts1, H1, shift1)

    # no need to warp image 2
    warp2, warp2_pts = im2, pts2

    print("Align and blend image 1 and 2")
    aligned1, aligned2, _, _, _, shift2 = rectification.align(
        warp1, warp2, warp1_pts, warp2_pts
    )
    utils.show_two(aligned1, aligned2)

    mosaic = rectification.blend(aligned1, aligned2, method=constants.BLEND_METHOD)
    plt.imshow(mosaic)
    return mosaic


def auto_stitch():
    im1, im2 = [utils.read_img(im, resize=True, gray=True) for im in args.images]
    points1, points2 = define_corners(im1, im2)

    points1 = np.flip(points1, axis=-1)
    points2 = np.flip(points1, axis=-1)

    im1, im2 = [utils.read_img(im, resize=True) for im in args.images]
    mosaic = stitch(im1, im2, points1, points2)

    mosaic_name = OUTDIR_2 / (name + "_mosaic.jpg")
    plt.imsave(mosaic_name, mosaic)
    print(f"Mosaic saved as {mosaic_name}")

    return


if __name__ == "__main__":

    num_imgs = len(args.images)
    name = (Path(args.images[0]).stem)[:-1]

    print(
        f"""
stitching {num_imgs} images for {name}...
detection: {args.detection}
"""
    )

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
        assert num_imgs == 2, num_imgs
        auto_stitch()
        sys.exit()
    else:
        raise ValueError()
