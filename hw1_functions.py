import cv2
import numpy as np
import matplotlib.pyplot as plt


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

    for k in range(256):
        d += pow(abs(hist1[k] - hist2[k]), 2)

    return pow(d, 0.5)


def meanSqrDist(im1, im2):
    return np.sum(np.power(im2.ravel() - im1.ravel(), 2)) / len(im1.ravel())


def sliceMat(im):
    # TODO: implement fucntion
    return Slices


def SLTmap(im1, im2):
    # TODO: implement fucntion
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
