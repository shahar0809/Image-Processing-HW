import numpy as np
import cv2.cv2 as cv2
import random
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Constants
MAX_GRAY_VAL = 256


def addSPnoise(im, p):
    sp_noise_im = im.copy()
    # Generate random indexes for white and black pixels
    white_pixels_x = random.sample(range(MAX_GRAY_VAL), int(p / 2))
    white_pixels_y = random.sample(range(MAX_GRAY_VAL), int(p / 2))
    black_pixels_x = random.sample(range(MAX_GRAY_VAL), int(p / 2))
    black_pixels_y = random.sample(range(MAX_GRAY_VAL), int(p / 2))

    # Assign white and black values
    sp_noise_im[white_pixels_x, white_pixels_y] = MAX_GRAY_VAL
    sp_noise_im[black_pixels_x, black_pixels_y] = 0
    return sp_noise_im


def addGaussianNoise(im, s):
    gaussian_noise_im = im.copy()
    # TODO: add implementation
    return gaussian_noise_im


def cleanImageMedian(im, radius):
    median_im = im.copy()
    # TODO: add implementation
    return median_im


def cleanImageMean(im, radius, maskSTD):
    # TODO: add implementation
    return cleaned_im


def bilateralFilt(im, radius, stdSpatial, stdIntensity):
    bilateral_im = im.copy()
    # TODO: add implementation
    return bilateral_im
