import cv2.cv2 as cv2
import numpy as np
from scipy.signal import convolve2d

BlACK = (0, 255, 0)


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


def canny_edge_detection(image, threshold1, threshold2, aperture_size=3, l2_gradient=False):
    return cv2.Canny(image, threshold1, threshold2, apertureSize=aperture_size, L2gradient=l2_gradient)


def hugh_transform_circles(image):
    blur_image = cv2.medianBlur(image, 9)
    circles = np.uint16(np.around(cv2.HoughCircles(blur_image, cv2.HOUGH_GRADIENT, 1, 15,
                                                   param1=59, param2=75, minRadius=2, maxRadius=0)))

    for circle in circles[0]:
        cv2.circle(image, (circle[0], circle[1]), circle[2], BlACK, 2)

    return image


def hugh_transform_lines(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=250)
    # Draw lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), BlACK, 3)
    # print(lines)
    # for line in lines[0]:
    #     a = np.cos(line[1])
    #     b = np.sin(line[1])
    #     x0 = a * line[0]
    #     y0 = b * line[0]
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
    #
    #     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return image
