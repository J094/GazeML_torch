"""Utility methods for generating and visualizing heatmaps."""
import numpy as np


def gaussian_2d(shape, centre, sigma=1.0):
    """Generate heatmap with single 2D gaussian."""
    # xs col
    xs = np.arange(0.5, shape[1] + 0.5, step=1.0, dtype=np.float32)
    # ys row
    ys = np.expand_dims(np.arange(0.5, shape[0] + 0.5, step=1.0, dtype=np.float32), -1)
    alpha = -0.5 / (sigma**2)
    heatmap = np.exp(alpha * ((xs - centre[0])**2 + (ys - centre[1])**2))
    return heatmap


def gaussian_2d_v2(shape, centre, sigma=1.0, visibility=2, scale=12):
    """Generate heatmap with single 2D gaussian."""
    height = int(shape[0])
    width = int(shape[1])
    heatmap = np.zeros(shape=(height, width))
    # this gaussian patch is 7x7, let's get four corners of it first
    x0 = int(centre[0])
    y0 = int(centre[1])
    xmin = int(x0 - 3 * sigma)
    ymin = int(y0 - 3 * sigma)
    xmax = int(x0 + 3 * sigma)
    ymax = int(y0 + 3 * sigma)
    if xmin >= width or ymin >= height or xmax < 0 or ymax <0 or visibility == 0:
        return heatmap

    size = int(6 * sigma + 1)
    x, y = np.meshgrid(np.arange(0, size, 1), np.arange(0, size, 1), indexing='xy')

    # the center of the gaussian patch should be 1
    centre_x = size // 2
    centre_y = size // 2
    # generate this 7x7 gaussian patch (Gaussian function in 2D)
    alpha = -0.5 / (sigma**2)
    gaussian_patch = np.exp(alpha * ((x - centre_x)**2 + (y - centre_y)**2)) * scale

    # part of the patch could be out of the boundary, so we need to determine the valid range
    # if xmin = -2, it means the 2 left-most columns are invalid, which is max(0, -(-2)) = 2
    patch_xmin = np.maximum(0, -xmin)
    patch_ymin = np.maximum(0, -ymin)

    # if xmin = 59, xmax = 66, but our output is 64x64, then we should discard 2 right-most columns
    # which is min(64, 66) - 59 = 5, and column 6 and 7 are discarded
    patch_xmax = np.minimum(xmax, width) - xmin
    patch_ymax = np.minimum(ymax, height) - ymin

    # also, we need to determine where to put this patch in the whole heatmap
    heatmap_xmin = np.maximum(0, xmin)
    heatmap_ymin = np.maximum(0, ymin)
    heatmap_xmax = np.minimum(xmax, width)
    heatmap_ymax = np.minimum(ymax, height)

    heatmap[heatmap_ymin:heatmap_ymax, heatmap_xmin:heatmap_xmax] = \
        gaussian_patch[patch_ymin:patch_ymax,patch_xmin:patch_xmax]
    return heatmap
