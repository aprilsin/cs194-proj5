import os
import pickle
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as io
from scipy.interpolate import interp2d
from scipy.spatial import Delaunay
from skimage import transform
from skimage.util import img_as_float, img_as_ubyte
import my_types as mt

ToImgArray = Union[str, Path, os.PathLike]


def read_img(x: ToImgArray, gray=False):
    assert isinstance(x, (str, Path, os.PathLike)), f"Didn't expect type {type(x)}"

    x = Path(x)
    assert x.suffix in (".jpeg", ".jpg"), x.suffix

    if gray:
        img = io.imread(x, as_gray=True)
    else:
        img = io.imread(x)
    return img_as_float(img)


def pick_points(img: ToImgArray, num_pts: int, APPEND_CORNERS=False) -> np.ndarray:
    """
    Returns an array of points for one image with ginput
    """
    img = read_img(img)
    print(f"Please select {num_pts} points in image.")
    plt.imshow(img)
    points = plt.ginput(num_pts, timeout=0)  # never timeout
    plt.close()

    if APPEND_CORNERS:
        y, x, _ = img.shape
        points.extend(
            [
                (0, 0),
                (0, y - 1),
                (x - 1, 0),
                (x - 1, y - 1),
            ]
        )
    assert len(points) == num_pts
    print(f"Picked {num_pts} points successfully.")
    return np.array(points)


def save_points(points: np.ndarray, name: os.PathLike) -> None:
    """
    Saves points as Pickle
    """
    name = Path(name)
    pickle_name = name.with_suffix(".pkl")
    pickle.dump(points, open(pickle_name, "wb"))


def load_points(name: os.PathLike) -> np.ndarray:
    """
    Loads an array of points saved as Pickle
    """
    name = Path(name)
    pickle_name = name.with_suffix(".pkl")
    return pickle.load(open(pickle_name, "rb"))


def plot_pts(img: np.ndarray, points: np.ndarray) -> None:
    """
    Displays the keypoints of an image
    """
    mt.assert_points(points)
    plt.figure()
    plt.imshow(img)
    plt.scatter(points[:, 0], points[:, 1], marker="o", color="b", s=30)
    plt.show()


def show_two(im1, im2):
    fig = plt.figure(figsize=(12, 12))
    p1 = fig.add_subplot(1, 2, 1)
    p1.imshow(im1)
    p2 = fig.add_subplot(1, 2, 2)
    p2.imshow(im2)
