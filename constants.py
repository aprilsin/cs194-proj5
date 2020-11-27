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

# gaussian filter
DEFAULT_KERNEL = 50
DEFAULT_SIGMA = 15

BLEND_METHOD = "average"
# BLEND_METHOD = "two-band"
# BLEND_METHOD = "alpha"


#####################
#      PART 2       #
#####################
MIN_HARRIS_STRENGTH = 0.1  # relative to max(image), i.e. most intense pixel of image
NUM_KEEP = 500  # want to keep the best 500 corners
MIN_RADIUS = 2
MAX_RADIUS = 500  # TODO should be 1/4 th of img size


@dataclass
# @total_ordering
class Corner:
    coord: np.ndarray  # shape = (2,  )
    strength: float


#     def __eq__(self, other):
#         return self.strength == other.strength

#     def __lt__(self, other):
#         return self.strength < other.strength

# #     def __hash__(self):
# #         return hash(self.coord) ^ hash(self.strength)


@dataclass
class Feature:
    coord: np.ndarray  # shape = (2,  )
    patch: np.ndarray  # shape = (8, 8)


# TODO
# combine the two data classes?
# ignore compare functions, manually implement them


class Tmp:
    def __init__(self, coord, strength, patch=None):
        # assert coord.ndim == 2, coord
        # assert len(coord) == 2, coord
        assert type(strength) == np.float64, type(strength)
        self.coord = coord
        self.strength = strength
        if patch is None:
            self.patch = np.zeros((8, 8))  # one patch is 8x8
        else:
            assert patch.shape == (8, 8), patch.shape
            self.patch = patch


combinations = {}
features = {}
