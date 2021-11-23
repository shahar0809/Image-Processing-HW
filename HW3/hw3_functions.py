import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def addSPnoise(im, p):
    sp_noise_im = im.copy()
	# TODO: add implementation
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


