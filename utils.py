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

import constants

ToImgArray = Union[np.ndarray, str, Path, os.PathLike]


def read_img(x: ToImgArray, resize=True, gray=False) -> np.ndarray:
    if isinstance(x, np.ndarray):
        img = img_as_float(x).clip(0, 1)
    elif isinstance(x, (str, Path, os.PathLike)):
        x = Path(x)
        if x.suffix in (".jpeg", ".jpg"):
            if gray:
                img = io.imread(x, as_gray=True)
            else:
                img = io.imread(x)
            img = img_as_float(img)
        else:
            raise ValueError(f"Didn't expect type {type(x)}")
    if resize:
        # resize image if too large
        num_pixels = 1600 * 1600
        h, w = img.shape[0], img.shape[1]
        r = int(h * w / num_pixels)
        if h * w > num_pixels and r > 1:
            return sk.transform.resize(
                img, (img.shape[0] // r, img.shape[1] // r), anti_aliasing=True
            )
    return img


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
    return np.array(points)  # in (x, y) format


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


def plot_points(img: np.ndarray, points: np.ndarray) -> None:
    """
    Displays the keypoints of an image
    points are in (x, y) format
    """
    fig = plt.figure()
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.scatter(x=points[:, 0], y=points[:, 1], marker="o", color="b", s=30)
    plt.show()
    return fig


def plot_corners(img: np.ndarray, points: np.ndarray) -> None:
    """
    Displays the keypoints of an image
    points are in (r, c) format
    """
    fig = plt.figure()
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.scatter(x=points[:, 1], y=points[:, 0], marker="o", color="b", s=30)
    plt.show()
    return fig


def show_two(im1, im2):
    fig = plt.figure(figsize=(12, 12))
    p1 = fig.add_subplot(1, 2, 1)
    p1.imshow(im1)
    p2 = fig.add_subplot(1, 2, 2)
    p2.imshow(im2)
    return fig


# handle file namings
def get_fname(name: str, i):
    pass


def to_gray(img: ToImgArray):
    img = read_img(img)
    if img.ndim == 3:
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        return (0.3 * R) + (0.59 * G) + (0.11 * B)
    return img


def vectorize(mat):
    assert mat.ndim == 2
    # h, w = mat.shape
    # return np.reshape(mat, (1, h * w))
    return mat.flatten()


def normalize(mat):
    return (mat - np.mean(mat)) / np.std(mat)


def dist2(
    x: Union[list, tuple, np.ndarray], c: Union[list, tuple, np.ndarray]
) -> np.ndarray:
    """
    dist2  Calculates squared distance between two sets of points.

    Input:
    - Takes two matrices of vectors and calculates the squared Euclidean distance between them.
    - Both matrices must be of the same column dimension.

    Output:
    - If X has M rows and N columns, and C has L rows and N columns, then the result has M rows and L columns.  The I, Jth entry is the squared distance from the Ith row of X to the Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """
    if isinstance(x, list):
        x = np.array(x)
    elif isinstance(x, tuple):
        x = np.array([list(x)])
    if isinstance(c, list):
        c = np.array(c)
    assert isinstance(x, np.ndarray) and isinstance(c, np.ndarray), (type(x), type(c))

    if constants.DEBUG:
        print("inputs: ", x.shape, c.shape)
    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, "Data dimension does not match dimension of centers"

    # dist^2 = r^2 + s^2 - 2*rs*cos(theta-phi)
    # dist^2 = r^2 + s^2 - 2*inner-product
    r_sq = np.ones((ncenters, 1)) * np.sum((x ** 2).T, axis=0)
    s_sq = np.ones((ndata, 1)) * np.sum((c ** 2).T, axis=0)

    sq_dist = (
        (np.ones((ncenters, 1)) * np.sum((x ** 2).T, axis=0)).T
        + np.ones((ndata, 1)) * np.sum((c ** 2).T, axis=0)
        - 2 * np.inner(x, c)
    )
    return sq_dist


def ssd_points(points1, points2):
    assert points1.ndim == points2.ndim == 2, (points1.shape, points2.shape)
    return np.sum((points1 - points2) ** 2, axis=-1)


# def dist_patches(patch1, patch2):
#     """
#     patch1 and patch2 are 8x8 grids.
#     """
#     assert patch1.shape == patch2.shape == (8, 8), (patch1.shape, patch2.shape)
#     patch1 = np.flatten(patch1)
#     patch2 = np.flatten(patch2)
#     return np.sum((patch1 - patch2) ** 2)  # sum squared distance
