import math
import random

import numpy as np
from scipy.signal import convolve2d


def addSPnoise(im, p):
    row, col = im.shape
    sp_noise_im = im.copy()
    num_noise_pixels = p * row * col

    # Generate random indexes for white and black pixels
    pixels_index_list = set(range(row * col))
    white_pixels_x = random.sample(pixels_index_list, int(num_noise_pixels / 2))
    black_pixels_x = random.sample(pixels_index_list - set(white_pixels_x), int(num_noise_pixels / 2))
    # Assign white and black values
    sp_noise_im.ravel()[white_pixels_x] = 255
    sp_noise_im.ravel()[black_pixels_x] = 0

    return sp_noise_im.reshape(im.shape)


def addGaussianNoise(im, s):
    noisy_image = im.copy()
    noisy_image = np.add(noisy_image, np.random.normal(0, s, im.shape))
    return noisy_image


def cleanImageMedian(im, radius):
    median_im = im.copy()
    for row in range(radius, im.shape[1] - radius):
        for col in range(radius, im.shape[0] - radius):
            filtering_mask = np.take(np.take(median_im, range(row - radius, row + radius + 1), 0),
                                     range(col - radius, col + radius + 1), 1)
            median_im[row][col] = np.median(filtering_mask)
    return median_im


def cleanImageMean(im, radius, maskSTD):
    col = list()
    for x in range(-radius, radius):
        row = list()
        for y in range(-radius, radius):
            row.append(math.exp(-((x ** 2 + y ** 2) / (2 * maskSTD ** 2))))
        col.append(row)
    mask = np.array(col)
    gaussian_filter = mask / np.sum(mask)
    cleaned_im = convolve2d(im, gaussian_filter, mode="same")
    return cleaned_im


def bilateralFilt(im, radius, stdSpatial, stdIntensity):
    bilateral_im = im.copy()
    # TODO: add implementation
    return bilateral_im
