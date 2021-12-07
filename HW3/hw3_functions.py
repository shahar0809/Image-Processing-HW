import math
import random
import matplotlib.pyplot as plt
import cv2
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
    for row in range(radius, im.shape[0] - radius):
        for col in range(radius, im.shape[1] - radius):
            filtering_mask = median_im[row - radius:row + radius + 1, col - radius:col + radius + 1]
            median_im[row][col] = np.median(filtering_mask)
    return median_im


def cleanImageMean(im, radius, maskSTD):
    mask = np.zeros((2 * radius, 2 * radius))
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            mask[x][y] = math.exp(-((x ** 2 + y ** 2) / (2 * maskSTD ** 2)))
    gaussian_filter = mask / np.sum(mask)
    cleaned_im = convolve2d(im, gaussian_filter, mode="same")
    return cleaned_im


def bilateralFilt(im, radius, stdSpatial, stdIntensity):
    bilateral_im = im.copy()

    for x in range(radius, im.shape[0] - radius):
        for y in range(radius, im.shape[1] - radius):
            window = bilateral_im[x - radius:x + radius + 1, y - radius:y + radius + 1].astype(float)

            gi = np.exp(-((window - im[x, y]) ** 2).astype(float) / np.float(2 * (stdIntensity ** 2)))
            gi = gi / np.sum(gi)

            xx, yy = np.meshgrid(list(range(x - radius, x + radius + 1)), list(range(y - radius, y + radius + 1)))
            gs = (np.exp(-((np.float_power(xx.ravel() - np.full(xx.ravel().shape, x), 2) + np.float_power(
                yy.ravel() - np.full(yy.ravel().shape, y), 2)) / (2 * stdSpatial ** 2)))).reshape(gi.shape)
            gs = gs.reshape(gi.shape).astype(float) / np.sum(gs)

            bilateral_im[x][y] = np.sum(cv2.multiply(cv2.multiply(gi, gs), window)) / np.sum(cv2.multiply(gi, gs))

    return bilateral_im.astype(np.uint8)
