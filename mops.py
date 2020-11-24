import filters
import itertools
import numpy as np
import utils
import numpy as np
import harris

def match_features(im1_grids, im2_grids):
    matched = []
    for i, j in itertools.product(len(im1_grids), len(im2_grids)):
        if np.equals(im1_grids[i], im2_grids[i]):
            matched.append([i, j])
    return matched