import cv2
import numpy as np
import matplotlib.pyplot as plt

GRAY_RANGE = 256

def print_IDs():
    # print("123456789")
    print("123456789+987654321\n")


def contrastEnhance(im, range):
    min_gray, max_gray = np.min(im), np.max(im)

    a = (np.max(range) - np.min(range)) / (max_gray - min_gray)
    b = np.max(range) - a * max_gray

    nim = a * im + b
    return nim, a, b


def showMapping(old_range, a, b):
    imMin = np.min(old_range)
    imMax = np.max(old_range)
    x = np.arange(imMin, imMax + 1, dtype=np.float)
    y = a * x + b
    plt.figure()
    plt.plot(x, y)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.title('contrast enhance mapping')


def minkowski2Dist(im1, im2):
    d = 0
    hist1 = np.histogram(im1, bins=256, range=(0, 255))
    hist2 = np.histogram(im2, bins=256, range=(0, 255))

    for gray_val in range(256):
        d += pow(abs(hist1[gray_val] - hist2[gray_val]), 2)

    return pow(d, 0.5)


def meanSqrDist(im1, im2):
    return np.sum(np.power(im2.ravel() - im1.ravel(), 2)) / len(im1.ravel())


def sliceMat(im):
    slices = np.zeros(np.size(im), 256)
    flat_im = im.ravel()

    for gray_val in (0, 256):
        slices[:, gray_val] = (flat_im == gray_val)

    return slices


def SLTmap(im1, im2):
    slice_mat = sliceMat(im1)

    return mapImage(im1, TM), TM


def mapImage(im, tm):
    # TODO: implement fucntion
    return TMim


def sltNegative(im):
    # TODO: implement fucntion - one line
    return nim


def sltThreshold(im, thresh):
    # TODO: implement fucntion
    return nim
