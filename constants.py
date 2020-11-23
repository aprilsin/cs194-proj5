from pathlib import Path

DATA = Path("data")
OUTDIR = Path("output")
OUTDIR_1 = Path("output/part1")
OUTDIR_2 = Path("output/part2")

assert DATA.exists()
assert OUTDIR.exists()
assert OUTDIR_1.exists()
assert OUTDIR_2.exists()

DEFAULT_KERNEL = 50
DEFAULT_SIGMA = 15

LOAD = False
SAVE = True
DEBUG = False

BLEND_METHOD = "average"
# BLEND_METHOD = "two-band"
# BLEND_METHOD = "alpha"
