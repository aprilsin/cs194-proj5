from pathlib import Path

DATA = Path("data")
OUTDIR_1 = Path("output_part1")
OUTDIR_2 = Path("output_part2")
assert DATA.exists()
assert OUTDIR_1.exists()
assert OUTDIR_2.exists()

NUM_PTS = 8

LOAD = True
SAVE = False
DEBUG = False

DEFAULT_KERNEL = 50
DEFAULT_SIGMA = 15

BLEND_METHOD = "average"
# BLEND_METHOD = "two-band"
# BLEND_METHOD = "alpha"
