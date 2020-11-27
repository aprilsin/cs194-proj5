from dataclasses import dataclass
from functools import total_ordering
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
OUTDIR_1 = Path("output_part1")
OUTDIR_2 = Path("output_part2")
assert DATA.exists()
OUTDIR_1.mkdir(exist_ok=True)
OUTDIR_2.mkdir(exist_ok=True)
# assert OUTDIR_1.exists()
# assert OUTDIR_2.exists()

#####################
#      PART 1       #
#####################

NUM_PTS = 8  # pick 8 points to define homography matrix


LOAD = False
SAVE = False
DEBUG = False
SHOW = False

# gaussian filter
DEFAULT_KERNEL = 5
DEFAULT_SIGMA = 0.8

BLEND_METHOD = "average"
# BLEND_METHOD = "two-band"
# BLEND_METHOD = "alpha"


#####################
#      PART 2       #
#####################

HARRIS_STRENGTH_THRESHOLD = (
    0.05  # relative to max(image), i.e. most intense pixel of image
)

NUM_KEEP = 500  # want to keep the best 500 corners

MIN_RADIUS = 1
MAX_RADIUS = 1600  # TODO should be 1/4 th of img size

MATCHING_THRESHOLD = 0.3

# select points at random (>=4 points are needed to compute homography)
NUM_SAMPLE_POINTS = 6

RANSAC_THRESHOLD = 0.2

colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
markers = ["o", ".", "+", "x", ">", "v", "^", "*", "D"]
