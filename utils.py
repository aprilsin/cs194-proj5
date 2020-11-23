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

ToImgArray = Union[np.ndarray, str, Path, os.PathLike]


def read_img(x: ToImgArray, gray=False) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return img_as_float(x).clip(0, 1)
    elif isinstance(x, (str, Path, os.PathLike)):
        x = Path(x)
        if x.suffix in (".jpeg", ".jpg"):
            img = io.imread(x)
            img = img_as_float(img)
            return img
        else:
            raise ValueError(f"Didn't expect type {type(x)}")


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
    assert len(points) == num_pts, len(points)
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


# handle file namings
def get_fname(name: str, i):
    pass
