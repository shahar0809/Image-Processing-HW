import cv2
import numpy as np
import matplotlib.pyplot as plt


def print_IDs():
    print("213991029")
    #print("213991029")


def contrastEnhance(im, range):
    # TODO: implement fucntion
    return nim, a,b


def showMapping(old_range, a, b):
    imMin = np.min(old_range)
    imMax = np.max(old_range)
    x = np.arange(imMin, imMax+1, dtype=np.float)
    y = a * x + b
    plt.figure()
    plt.plot(x, y)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.title('contrast enhance mapping')


def minkowski2Dist(im1,im2):
    # TODO: implement fucntion
    return d


def meanSqrDist(im1, im2):
    # TODO: implement fucntion - one line
    return d


def sliceMat(im):
    # TODO: implement fucntion
    return Slices


def SLTmap(im1, im2):
    # TODO: implement fucntion
    return mapImage(im1, TM), TM


def mapImage (im,tm):
    # TODO: implement fucntion
    return TMim


def sltNegative(im):
    # TODO: implement fucntion - one line
    return nim


def sltThreshold(im, thresh):
    # TODO: implement fucntion
    return nim
