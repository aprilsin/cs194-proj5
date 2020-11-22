import math
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp
import skimage as sk
import skimage.io as skio
import skimage.transform as sktr
from numpy import ndarray
from PIL import Image, ImageOps
from scipy import signal
from skimage import img_as_float, img_as_ubyte
from sklearn import preprocessing
from tqdm import tqdm, trange


def gaussian_filter(kernel_size, sigma, display=False):
    """returns a 2D gaussian filter"""
    gauss_1d = cv2.getGaussianKernel(kernel_size, sigma)
    gauss_2d = gauss_1d.T * gauss_1d
    if display:
        plt.figure()
        plt.imshow(gauss_2d)
    return gauss_2d


def gauss_blur(im, kernel_size=15, sigma=3, display=False):
    """ blurrs an image"""
    assert np.ndim(im) == 3

    blur_channels = []
    gauss_filter = gaussian_filter(kernel_size, sigma)

    for ch in range(3):
        channel = im[:, :, ch]
        low_freq = signal.convolve2d(channel, gauss_filter, mode="same")
        blur_channels.append(low_freq)

    result = np.stack(blur_channels, axis=-1)

    if display:
        fig = plt.figure(figsize=(12, 12))
        p1 = fig.add_subplot(1, 2, 1)
        p1.imshow(im)
        p1.set_title("input")
        p2 = fig.add_subplot(1, 2, 2)
        p2.imshow(result)
        p2.set_title("blurred")
        pass

    return result


def unsharp_mask(im, kernel_size=15, sigma=3, display=False):
    """sharpens an image by amplifying high frequencies"""
    assert np.ndim(im) == 3

    sharpened_channels = []
    gauss_filter = gaussian_filter(kernel_size, sigma)

    for ch in range(3):
        channel = im[:, :, ch]
        low_freq = signal.convolve2d(channel, gauss_filter, mode="same")
        high_freq = channel - low_freq
        sharpened = channel + high_freq
        sharpened_channels.append(sharpened)

    result = np.stack(sharpened_channels, axis=-1)

    if display:
        fig = plt.figure(figsize=(12, 12))
        p1 = fig.add_subplot(1, 2, 1)
        p1.imshow(im)
        p1.set_title("input")
        p2 = fig.add_subplot(1, 2, 2)
        p2.imshow(result)
        p2.set_title("sharpened")
        pass

    return result


def gaussian_stack(im, gauss_filter, num_levels=5):
    assert im.ndim == gauss_filter.ndim
    stack = []

    def g_stack(prev_level, level):
        if level == 0:
            return
        else:
            new_level = signal.convolve2d(prev_level, gauss_filter, mode="same")
            stack.append(new_level)
            g_stack(new_level, level - 1)

    g_stack(im, num_levels)
    return stack


def laplacian_stack(gauss_stack):
    Lstack = []
    for i in range(len(gauss_stack)):
        Lstack.append(gauss_stack[i - 1] - gauss_stack[i])
    return Lstack
