from dataclasses import dataclass
from functools import total_ordering
from pathlib import Path

import numpy as np
import pandas as pd

#####################
#       PATHS       #
#####################

DATA = Path("data")
# assert DATA.exists()

OUTDIR_1a = Path("output_part1/to_plane")
OUTDIR_1b = Path("output_part1/to_middle")
OUTDIR_2a = Path("output_part2/define_corners")
OUTDIR_2b = Path("output_part2/stitching")
# OUTDIR_1a.mkdir(exist_ok=True)
# OUTDIR_1b.mkdir(exist_ok=True)
# OUTDIR_2.mkdir(exist_ok=True)

DEBUG = None
SHOW = None


#####################
#      PART 1       #
#####################

NUM_PTS = 8  # pick 8 points to define homography matrix

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
    0.1  # relative to max(image), i.e. most intense pixel of image
)

NUM_KEEP = 500  # want to keep the best 500 corners

MIN_RADIUS = 2
MAX_RADIUS = 1600

MATCHING_THRESHOLD = 0.3

# select points at random (>=4 points are needed to compute homography)
NUM_SAMPLE_POINTS = 6

RANSAC_THRESHOLD = 3  # 3 pixels


#####################
#      OTHERS       #
#####################

MARKER_SIZE = 25
colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
markers = ["o", ".", "+", "x", ">", "v", "^", "*", "D"]
# {time.time():.0f}


@dataclass
class Corner:
    coord: np.ndarray
    strength: np.float64
    patch: np.ndarray = None
