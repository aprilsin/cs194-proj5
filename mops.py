import filters, harris


def detect_corners(img):
    # num_levels = 5
    # g_stack = filters.gaussian_stack(img)
    # for level in g_stack:
    assert img.ndim == 2, img.shape
    blurred = filters.gauss_blur(img)
    harris_stength, coords = harris.get_harris_corners(blurred)


def corner_strength():
    """the corner strength function"""
    pass


def standardize():
    """make all detected corner an 8x8 grid"""
    pass
