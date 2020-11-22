import filters


def detect_corners(img):
    # num_levels = 5
    # g_stack = filters.gaussian_stack(img)
    # for level in g_stack:
    blurred = filters.gauss_blur(img)
    gradient_mag, gradient_angles = filters.getGradient(img)


def corner_strength():
    """the corner strength function"""
    pass


def standardize():
    """make all detected corner an 8x8 grid"""
    pass