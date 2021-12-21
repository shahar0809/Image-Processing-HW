import cv2.cv2 as cv2
from scipy.signal import convolve2d
import numpy as np

def sobel_edge_detection(img):
    sobel_vertical_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_horizontal_kernel = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    vertical_edges = convolve2d(img, sobel_vertical_kernel)
    return convolve2d(vertical_edges, sobel_horizontal_kernel)
