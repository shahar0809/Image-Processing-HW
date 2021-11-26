import numpy as np
import cv2.cv2 as cv2
import random
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Constants
MAX_GRAY_VAL = 256


def addSPnoise(im, p):
    row, col = im.shape
    sp_noise_im = im.copy()
    num_noise_pixels = p * row * col

    # Generate random indexes for white and black pixels
    white_pixels_x = random.sample(range(row * col), int(num_noise_pixels / 2))
    black_pixels_x = random.sample(range(row * col), int(num_noise_pixels / 2))

    # Assign white and black values
    sp_noise_im.ravel()[white_pixels_x] = 255
    sp_noise_im.ravel()[black_pixels_x] = 0

    return sp_noise_im.reshape(im.shape)


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