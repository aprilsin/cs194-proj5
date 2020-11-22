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
import my_types
import rectification
import utils

parser = argparse.ArgumentParser()

parser.add_argument(
    "detection",
    type=str,
    choices=["manual", "auto"],
    default="manual",
    help="Choose manual or auto stitching",
)
# parser.add_argument("stitch", type=str, choices=["plane", "merge"])
parser.add_argument("images", nargs="+")
parser.add_argument(
    "-s",
    "--save",
    type=bool,
    default=False,
    help="Whether intermediate data will be saved.",
)

args = parser.parse_args()


def manual_stitching():
    pass


def auto_stitching():
    pass


if __name__ == "__main__":

    num_imgs = len(args.images)
    name = (Path(args.images[0]).stem)[:-1]

    print(
        f"""
stitching {num_imgs} images for {name}...
    detection-method = {args.detection}
    stitching-method = {args.stitch}
    """
    )

    # set up file names
    DATA = Path("data")
    OUTDIR = Path("output")
    assert DATA.exists()
    assert CACHE.exists()
    assert OUTDIR.exists()
    (DATA / name).mkdir(parents=False, exist_ok=True)

    # read images to arrays
    imgs = [utils.reag_img(im) for im in args.images]

    if args.detection == "manual":
        if num_imgs == 2:

            pts = [utils.pick_points(im, 4) for im in imgs]

            h, w, c = imgs[0].shape
            num_pixels = 1600 * 1600
            RESIZE = h * w > num_pixels
            if RESIZE:
                print("resizing to target shape ({h / r}, {w / r}")
                imgs = [
                    sk.transform.resize(
                        im, (im.shape[0] // r, im.shape[1] // r), anti_aliasing=True
                    )
                    for im in imgs
                ]
                pts = [p / r for p in pts]
        elif num_imgs == 3:

            pass
        else:
            raise ValueError(f"{num_imgs = }")
