# This file contains masking functions extracted from niftynet
from __future__ import absolute_import, print_function, division

import numpy as np
import scipy.ndimage as ndimg
from scipy.ndimage.morphology import binary_fill_holes as fill_holes

def otsu_threshold(img, nbins=256):
    """
    Implementation of otsu thresholding

    :param img:
    :param nbins:
    :return:
    """
    hist, bin_edges = np.histogram(img.ravel(), bins=nbins)
    hist = hist.astype(float)
    half_bin_size = (bin_edges[1] - bin_edges[0]) * 0.5
    bin_centers = bin_edges[:-1] + half_bin_size

    weight_1 = np.copy(hist)
    mean_1 = np.copy(hist)
    weight_2 = np.copy(hist)
    mean_2 = np.copy(hist)
    for i in range(1, hist.shape[0]):
        weight_1[i] = weight_1[i - 1] + hist[i]
        mean_1[i] = mean_1[i - 1] + hist[i] * bin_centers[i]

        weight_2[-i - 1] = weight_2[-i] + hist[-i - 1]
        mean_2[-i - 1] = mean_2[-i] + hist[-i - 1] * bin_centers[-i - 1]

    target_max = 0
    threshold = bin_centers[0]
    for i in range(0, hist.shape[0] - 1):
        ratio_1 = mean_1[i] / weight_1[i]
        ratio_2 = mean_2[i + 1] / weight_2[i + 1]
        target = weight_1[i] * weight_2[i + 1] * (ratio_1 - ratio_2) ** 2
        if target > target_max:
            target_max, threshold = target, bin_centers[i]
    return threshold

def make_mask_3d(image, thr = 0.5, type_str = "mean_plus"):
    assert image.ndim == 3
    image_shape = image.shape
    image = image.reshape(-1)
    mask = np.zeros_like(image, dtype=np.bool)
    if type_str == 'threshold_plus':
        mask[image > thr] = True
    elif type_str == 'threshold_minus':
        mask[image < thr] = True
    elif type_str == 'otsu_plus':
        thr = otsu_threshold(image) if np.any(image) else thr
        mask[image > thr] = True
    elif type_str == 'otsu_minus':
        thr = otsu_threshold(image) if np.any(image) else thr
        mask[image < thr] = True
    elif type_str == 'mean_plus':
        thr = np.mean(image)
        mask[image > thr] = True
    mask = mask.reshape(image_shape)
    mask = ndimg.binary_dilation(mask, iterations=2)
    mask = fill_holes(mask)
    # foreground should not be empty
    assert np.any(mask == True), \
        "no foreground based on the specified combination parameters, " \
        "please change choose another `mask_type` or double-check all " \
        "input images"
    return mask


# This function returns the bbox of a mask
def get_mask_bbox_3D(mask):
    x_idx, y_idx, z_idx = np.where(mask>0)
    xmin, xmax = np.min(x_idx), np.max(x_idx)
    ymin, ymax = np.min(y_idx), np.max(y_idx)
    zmin, zmax = np.min(z_idx), np.max(z_idx)
    bbox = np.array([xmin, xmax, ymin, ymax, zmin, zmax])
    
    return bbox
