import cv2.cv2 as cv2
from scipy.signal import convolve2d
import numpy as np

def sobel_edge_detection(img):
    sobel_vertical_kernel = np.matrix([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], np.float32)

    sobel_horizontal_kernel = np.matrix([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], np.float32)

    vertical_edges = convolve2d(img, sobel_vertical_kernel)
    horizontal_edges = convolve2d(img, sobel_horizontal_kernel)
    sobel_edges = np.hypot(vertical_edges, horizontal_edges)

    # Normalize back to range of [0, 255]
    sobel_edges = sobel_edges / sobel_edges.max() * 255
    return sobel_edges

def threshold_filter(img, threshold):
    img_clone = img.copy()
    img_clone[img >= threshold] = 255
    img_clone[img < threshold] = 0
    return img_clone
