import cv2.cv2 as cv2
import numpy as np
from scipy.signal import convolve2d

BlACK = (0, 255, 0)
RED = (255, 0, 0)

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
    # return sobel_edges, np.arctan2(vertical_edges, horizontal_edges) * (180 / np.pi) % 360
    return sobel_edges, np.arctan2(vertical_edges, horizontal_edges) * (180 / np.pi) % 360

def threshold_filter(img, threshold):
    img_clone = img.copy()
    img_clone[img >= threshold] = 255
    img_clone[img < threshold] = 0
    return img_clone


def canny_edge_detection(image, threshold1, threshold2, aperture_size=3, l2_gradient=False):
    return cv2.Canny(image, threshold1, threshold2, apertureSize=aperture_size, L2gradient=l2_gradient)


def hough_transform_circles(image):
    blur_image = cv2.medianBlur(image, 9)
    circles = np.uint16(np.around(cv2.HoughCircles(blur_image, cv2.HOUGH_GRADIENT, 1, 15,
                                                   param1=59, param2=75, minRadius=2, maxRadius=0)))

    for circle in circles[0]:
        cv2.circle(image, (circle[0], circle[1]), circle[2], BlACK, 2)

    return image


def hough_transform_lines1(image):
    edges = cv2.Canny(image, 40, 170, apertureSize=3, L2gradient=True)
    return draw_lines(image, cv2.HoughLinesP(edges, 1, np.pi / 340, 197, minLineLength=190, maxLineGap=30))

def hough_transform_lines2(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    return draw_lines(image, cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=250))

def hough_transform_lines3(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    return draw_lines(image, cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=250))

def draw_lines(image, lines):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), BlACK, 3)

    return image
