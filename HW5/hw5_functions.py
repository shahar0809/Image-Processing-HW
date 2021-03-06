import cv2.cv2 as cv2
import numpy as np
from scipy.signal import convolve2d

BlACK = (0, 0, 0)
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


def canny_edge_detection1(image):
    kernel = np.ones((5, 5), np.float32) / 25
    blur_image = cv2.filter2D(image, -1, kernel)
    return cv2.Canny(blur_image, 53, 203, L2gradient=True)


def canny_edge_detection2(image):
    blur_image = cv2.medianBlur(image, 5)
    return cv2.Canny(blur_image, 20, 95, L2gradient=True)


def hough_transform_circles(image):
    blur_image = cv2.medianBlur(image, 9)
    circles = np.uint16(np.around(cv2.HoughCircles(blur_image, cv2.HOUGH_GRADIENT, 1, 15,
                                                   param1=59, param2=75, minRadius=2, maxRadius=0)))

    for circle in circles[0]:
        cv2.circle(image, (circle[0], circle[1]), circle[2], BlACK, 2)

    return image


def hough_transform_lines1(image):
    blur_image = cv2.medianBlur(image, 5)
    edges = cv2.Canny(blur_image, 170, 296, apertureSize=3)
    lines = cv2.HoughLines(edges, 2, 0.6 * (np.pi / 180), 250)
    lines = merge_lines(lines, 0.5, 6)
    return draw_lines_polar(image, lines)


def hough_transform_lines2(image):
    edges = cv2.Canny(image, 350, 400, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, 2 * np.pi / 180, 86)
    lines = merge_lines(lines, 0.5, 5)
    return draw_lines_polar(image, lines, 4)


def hough_transform_lines3(image):
    blur_image = cv2.GaussianBlur(image, (35, 35), 1)
    edges = cv2.Canny(blur_image, 10, 170, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, 3 * np.pi / 180, 122)
    lines = np.squeeze(lines)
    return draw_lines_polar(image, lines, 1)


"""
Utility functions for removing similar lines.
"""


def get_points(line):
    r, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * r
    y0 = b * r
    x1 = int(x0 + 1000 * -b)
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * -b)
    y2 = int(y0 - 1000 * a)
    return np.array([x1, y1]), np.array([x2, y2])


def get_line_equation(line):
    point1, point2 = get_points(line)

    if point2[0] == point1[0]:
        slope = (point2[1] - point1[1]) / (point2[0] - point1[0] + 0.001)
    else:
        slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    return slope, point1[1] - slope * point1[0]


def dist_between_lines(line1, line2):
    coefficients1 = get_line_equation(line1)
    coefficients2 = get_line_equation(line2)

    slope1 = coefficients1[0]
    b1 = coefficients1[1]

    slope2 = coefficients2[0]
    b2 = coefficients2[1]
    return abs(b1 - b2) / np.sqrt(np.square(slope1) + np.square(slope2))


def merge_lines(lines, slope_thresh, b_thresh):
    merged_lines = np.squeeze(lines)
    for line1 in np.squeeze(lines):
        for line2 in np.squeeze(lines):
            if np.any(np.not_equal(line1, line2)):
                l1 = get_line_equation(line1)
                l2 = get_line_equation(line2)
                if abs(l1[0] - l2[0]) < slope_thresh and dist_between_lines(line1, line2) < b_thresh:
                    merged_lines = merged_lines[np.not_equal(merged_lines, line2)[:, 0]]
    return list(merged_lines)


def draw_lines_polar(image, lines, thickness=2):
    for line in lines:
        point1, point2 = get_points(line)
        cv2.line(image, (point1[0], point1[1]), (point2[0], point2[1]), BlACK, thickness)

    return image


def draw_lines(image, lines, thickness=2):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), BlACK, thickness)
    return image
